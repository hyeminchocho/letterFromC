class ConditionalBatchNorm extends tf.layers.Layer{

    constructor(name, cbn_idx, conditioning_vector, k_reg){
        // super(ConditionalBatchNorm, self).__init__(name='CBN' + '_' + name + '_' + str(cbn_idx));
        super({name:'CBN_'+name+'_'+cbn_idx});
        this.conditioning_vector = conditioning_vector;
        this.k_reg = k_reg;
    }

    call(inputs, kwargs){
        let net = tf.layers.batchNormalization({scale:false, center:false}).apply(inputs);
        let num_channels = net.shape[-1];

        let gamma = tf.layers.dense({units:num_channels, use_bias:false, activation:'linear',
                                     kernel_regularizer:this.k_reg,
                                     kernel_initializer:tf.initializers.orthogonal()}).apply(this.conditioning_vector);
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels]);
        net *= gamma;

        let beta = tf.layers.dense({units:num_channels, use_bias:false,
                                    activation:'linear', kernel_regularizer:this.k_reg,
                                    kernel_initializer:tf.initializers.orthogonal()}).apply(this.conditioning_vector);
        beta = tf.reshape(beta, [-1, 1, 1, num_channels]);
        net += beta;
        return net;
    }

    getClassName() { return 'ConditionalBatchNorm';}
}

class ResNetBlockUp extends tf.layers.Layer{
    constructor(config){
        super({name:'ResNetBlockUp' + '_' + config.name});
        this.nm = config.name;
        this.output_dim = config.output_dim;
        this.is_last_block = config.is_last_block;
        this.conditioning_vector = config.conditioning_vector;
        this.k_reg = config.k_reg;
    }

    call(inputs, kwargs){
        let net = inputs;

        net = ConditionalBatchNorm({name:this.nm, cbn_idx:1, conditioning_vector:this.conditioning_vector,
                                    k_reg:this.k_reg}).apply(net);
        net = tf.nn.relu(net);

        if (this.is_last_block){
            up_stride = (2, 1);
        } else {
            up_stride = (2, 2);
        }

        net = tf.layers.conv2DTranspose({filters:this.output_dim, kernelSize:(3, 3), strides:up_stride,
                                         kernel_regularizer:this.k_reg,
                                         kernel_initializer:tf.initializers.orthogonal(),
                                         padding:'same', use_bias:true}).apply(net);

        net = ConditionalBatchNorm({name:this.nm, cbn_idx:2, conditioning_vector:this.conditioning_vector,
                                    k_reg:this.k_reg}).apply(net);
        net = tf.nn.relu(net);
        net = tf.layers.conv2D({filter:this.output_dim, kernelSize:(3, 3), strides:(1, 1),
                                kernelInitializer:tf.initializers.orthogonal(),
                                kernelRegularizer:this.k_reg, padding:'same', use_bias:true}).apply(net);

        shortcut = tf.layers.conv2DTranspose({filters:this.output_dim, kernelSize:(1, 1), strides:up_stride,
                                              kernel_initializer:tf.initializers.orthogonal(),
                                              kernel_regularizer:this.k_reg,
                                              padding:'same', use_bias:True}).apply(inputs);
        net += shortcut;
        return net;
    }

    getClassName() { return 'ResNetBlockUp';}
}

class NonLocalBlock extends tf.layers.Layer{
    constructor(name, k_reg){
        super({name:'NonLocalBlock' + '_' + name});
        this.k_reg = k_reg;
    }

    build(input_shape){
        this.sigma = this.addWeight({name:"sigma",
                                     shape:[],
                                     initializer:'zeros',
                                     trainable:true});
    }

    _spatial_flatten(inputs){
        shape = inputs.shape;
        return tf.reshape(inputs, (tf.shape(inputs)[0], -1, shape[3]));
    }


    call(input){
        let h = input.get_shape().as_list()[1];
        let w = input.get_shape().as_list()[2];
        let num_channels = input.get_shape().as_list()[3];

        let num_channels_attn = Math.floor(num_channels / 8);
        let num_channels_g = Math.floor(num_channels / 2);

        let theta = tf.layers.conv2D({filters:num_channels_attn, kernel_size:(1, 1), use_bias:false,
                                strides:(1, 1),
                                padding:'same', kernel_initializer:tf.initializers.orthogonal(),
                                  kernel_regularizer:this.k_reg, name:"conv2d_theta"}).apply(input);
        theta = this._spatial_flatten(theta);

        let phi = tf.layers.conv2D({filters:num_channels_attn, kernel_size:(1, 1), use_bias:false,
                                    strides:(1, 1),
                                    padding:'same', kernel_initializer:tf.initializers.orthogonal(),
                                    kernel_regularizer:this.k_reg, name:"conv2d_phi"}).apply(input);
        phi = tf.layers.maxPooling2d({poolSize:[2, 2], strides:2}).apply(phi);
        phi = this._spatial_flatten(phi);

        let attn = tf.matMul(theta, phi, false, true);
        attn = tf.nn.softmax(attn);

        let g = tf.layers.conv2D({filters:num_channels_g, kernelSize:(1, 1), useBias:false,
                                  strides:(1, 1),
                                  padding:'same', kernelInitializer:tf.initializers.orthogonal(),
                                  kernelRegularizer:this.k_reg, name:"conv2d_g"}).apply(input);
        g = tf.layers.maxPooling2d({poolSize:[2, 2], strides:2}).apply(g);
        g = this._spatial_flatten(g);

        let attn_g = tf.matmul(attn, g);
        attn_g = tf.reshape(attn_g, [tf.shape(attn_g)[0], h, -1, num_channels_g]);
        attn_g = layers.Conv2D({filters:num_channels, kernelSize:(1, 1), useBias:false, strides:(1, 1),
                               padding:'same', kernelInitializer:tf.initializers.orthogonal(),
                               kernelRegularizer:this.k_reg, name:"conv2d_attn_g"}).apply(attn_g);

        return this.sigma * attn_g + input;
    }

    getClassName() { return 'NonLocalBlock'; }

}

class SpatialEmbedding extends tf.layers.Layer{
    constructor(config){
        super({name:'SpatialEmbedding'});
        this.vocab_size = config.vocab_size;
        this.filter_dim = config.filter_dim;
    }

    build(inputShape){
        this.kernel = this.addWeight("filter_bank",
                                     [this.vocab_size, this.filter_dim[0], this.filter_dim[1]],
                                     null, tf.initializers.zeros(), null, true);
    }

    computeOutputShape(inputShape){
        return [inputShape[0], this.filter_dim[0], this.filter_dim[1]];
    }

    call(inputs, kwargs){
        // this.invokeCallHook(inputs, kwargs);
        // let a = tf.randomNormal([this.vocab_size, this.filter_dim[0], this.filter_dim[1]]);
        // return tf.gatherND(a, inputs);
        // let gat = tf.gatherND(this.kernel.val, inputs);
        let gat = tf.gather(this.kernel.val, inputs, 0);
        return gat;
    }
    getClassName() { return 'SpatialEmbedding'; }

}

class GeneratorPrep extends tf.layers.Layer{
    constructor(config){
        super({name:'GeneratorPrep'});
        this.y = config.y;
        this.num_blocks = config.num_blocks;
        this.vocab_size = config.vocab_size;
        this.seed_size_h = config.seed_size_h;
        this.seed_size_w = config.seed_size_w;
        this.embed_y = config.embed_y;
        this.out_channels = config.out_channels;
        this.kernel_reg = config.kernel_reg;
        this.blocks_with_attention = config.blocks_with_attention;
    }

    // call(z, y, num_blocks, vocab_size, embed_y, kwargs){
    call(inputs, kwargs){
        let spatial_embedding = new SpatialEmbedding({vocab_size:this.vocab_size, filter_dim:this.embed_y});
        let z = inputs[0];
        let y = inputs[1].asType('int32');
        // let se_layer = spatial_embedding.apply(this.y);
        let se_layer = spatial_embedding.apply(y);

        let z_per_block = tf.split(z, this.num_blocks + 1, 1);
        let z0 = z_per_block[0];
        // z0 = tf.reshape(z0, (-1, 1, 1, z0.shape[1]));
        z0 = z0.reshape([-1, 1, 1, z0.shape[1]]);

        // let net = tf.linalg.matmul(tf.tile(z0, [1, tf.shape(se_layer)[1], 1, 1]), se_layer);
        let net = tf.tile(z0, [1, se_layer.shape[0], 1, 1]);
        // net = net.matMul(se_layer);
        console.log("lalalalala");
        console.log(z0);
        console.log(se_layer);
        console.log(net);
        net = se_layer.matMul(net);
        net = tf.squeeze(net, 2);

        net = tf.reshape(net, (tf.shape(net)[0], 512, this.seed_size_h, this.seed_size_w, -1));
        net = tf.reshape(net, (tf.shape(net)[0], -1, 512, this.seed_size_h));
        net = tf.transpose(net, (0, 3, 1, 2));

        for (let block_idx = 0; block_idx < this.num_blocks; block_idx++){
            let name = "B"+(block_idx + 1).toString();
            let is_last_block = block_idx == this.num_blocks - 1;
            net = ResNetBlockUp({name:name, output_dim:this.out_channels[block_idx], is_last_block:is_last_block,
                                 conditioning_vector:z_per_block[block_idx], k_reg:this.kernel_reg}).apply(net);
            if (name in this.blocks_with_attention){
                net = NonLocalBlock(name, this.kernel_reg)(net);
            }
        }
        net = tf.layers.batchNormalization().apply(net);
        net = tf.relu(net);
        net = tf.layers.conv2D({filter:c, kernelSize:(3, 3),
                                strides:(1, 1),
                                kernelInitializer:tf.initializers.orthogonal(),
                                kernelRegularizer:this.kernel_reg,
                                padding:'same'}).apply(net);

        net = tf.tanh(net);
        return net;
    }

    getClassName() { return 'GeneratorPrep';}
}


function makeGenerator(latent_dim, input_dim, embed_y, gen_path, kernel_reg, blocks_with_attention,
                       vocab_size, vis_model){

    let h = input_dim[0];
    let c = input_dim[2];

    // in_channels, out_channels = get_in_out_channels_gen(h);
    let ch = 64;
    let in_channels = [ch*8, ch*4, ch*2];
    let out_channels = [ch*4, ch*2, ch*1];
    let num_blocks = 3;

    let seed_size_h = Math.round(h / 2 ** num_blocks);
    let seed_size_w = seed_size_h;

    let z = tf.input({shape:[latent_dim]});
    let y = tf.input({shape:[null], dtype:tf.int32});

    let genPrep = new GeneratorPrep({y:y, num_blocks:num_blocks, vocab_size:vocab_size,
                                     seed_size_h:seed_size_h, seed_size_w:seed_size_w,
                                     out_channels:out_channels,
                                     kernel_reg:kernel_reg,
                                     blocks_with_attention:blocks_with_attention,
                                     embed_y:embed_y});
    let net = genPrep.apply([z, y]);
    // let z_per_block = genPrep.z_per_block;


    // model = tf.keras.Model([z, y], net);
    let model = tf.model({inputs:[z, y], outputs:net});

    return model;
}
let seed = tf.randomNormal([1, 128]);
let labels = tf.tensor([0, 1, 2], undefined, 'int32');
let g = makeGenerator(128, [32, 160, 1], [32, 8192], '', 'spectral_norm', 'B3', 52, false);
g.apply([seed, labels]);
