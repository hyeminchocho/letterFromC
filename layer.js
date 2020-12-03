let n = new npyjs();

class ConditionalBatchNorm extends tf.layers.Layer{

    constructor(config){
        // super(ConditionalBatchNorm, self).__init__(name='CBN' + '_' + name + '_' + str(cbn_idx));
        super({name:'CBN_'+config.name+'_'+config.cbn_idx});
        this.conditioning_vector = config.conditioning_vector;
        this.k_reg = config.k_reg;
    }

    call(inputs, kwargs){
        this.bn = tf.layers.batchNormalization({scale:false, center:false});
        let net = this.bn.apply(inputs);
        let num_channels = net.shape[3];

        this.dense1 = tf.layers.dense({units:num_channels, useBias:false, activation:'linear',
                                       kernelRegularizer:tf.regularizers.l2({}),
                                       kernelInitializer:tf.initializers.orthogonal({gain:1})});
        let gamma = this.dense1.apply(this.conditioning_vector);
        gamma = gamma.reshape([-1, 1, 1, num_channels]);
        gamma = tf.tile(gamma, [1, net.shape[1], net.shape[2], 1]);
        // net *= gamma;
        net = net.mul(gamma);

        this.dense2 = tf.layers.dense({units:num_channels, useBias:false,
                                       activation:'linear', kernelRegularizer:tf.regularizers.l2({}),
                                       kernelInitializer:tf.initializers.orthogonal({gain:1})});
        let beta = this.dense2.apply(this.conditioning_vector);
        beta = beta.reshape([-1, 1, 1, num_channels]);
        // net += beta;
        net = net.add(beta);
        return net;
    }

    computeOutputShape(inputShape){
        return inputShape;
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

        this.cbn = new ConditionalBatchNorm({name:this.nm, cbn_idx:1,
                                        conditioning_vector:this.conditioning_vector,
                                        k_reg:this.k_reg});
        net = this.cbn.apply(net);
        net = tf.relu(net);

        let up_stride = (2, 2);
        if (this.is_last_block){
            up_stride = (2, 1);
        }

        this.conv2dT1 = tf.layers.conv2dTranspose({filters:this.output_dim, kernelSize:(3, 3), strides:up_stride,
                                                  kernelRegularizer:tf.regularizers.l2(),
                                                  kernelInitializer:tf.initializers.randomNormal({mean:0.0,
                                                                                                  stddev:1.0}),
                                                  padding:'same', useBias:true});
        net = this.conv2dT1.apply(net);

        this.cbn2 = new ConditionalBatchNorm({name:this.nm, cbn_idx:2,
                                         conditioning_vector:this.conditioning_vector,
                                         k_reg:this.k_reg});
        net = this.cbn2.apply(net);
        net = tf.relu(net);
        this.conv2d = tf.layers.conv2d({filters:this.output_dim, kernelSize:(3, 3), strides:(1, 1),
                                        kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                        kernelRegularizer:tf.regularizers.l2(), padding:'same', useBias:true});
        net = this.conv2d.apply(net);

        this.conv2dT2 = tf.layers.conv2dTranspose({filters:this.output_dim, kernelSize:(1, 1), strides:up_stride,
                                                   kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                                   kernelRegularizer:tf.regularizers.l2(),
                                                   padding:'same', useBias:true});
        let shortcut = this.conv2dT2.apply(inputs);
        net = net.add(shortcut);
        return net;
    }

    computeOutputShape(inputShape){
        return [inputShape[0], inputShape[1]*2, inputShape[2]*2, inputShape[3]/2];
    }

    getClassName() { return 'ResNetBlockUp';}
}

class NonLocalBlock extends tf.layers.Layer{
    constructor(config){
        super({name:'NonLocalBlock' + '_' + config.name});
        this.k_reg = config.k_reg;
    }

    build(input_shape){
        // this.sigma = this.addWeight({name:"sigma",
        //                              shape:[],
        //                              initializer:'zeros',
        //                              trainable:true});
        this.sigma = this.addWeight("sigma",
                                     [],
                                    'float32',
                                    tf.initializers.zeros(),
                                    undefined,
                                     true);
    }

    _spatial_flatten(inputs){
        let shape = inputs.shape;
        return inputs.reshape([inputs.shape[0], -1, shape[3]]);
    }


    call(input){
        let h = input.shape[1];
        let w = input.shape[2];
        let num_channels = input.shape[3];

        let num_channels_attn = Math.floor(num_channels / 8);
        let num_channels_g = Math.floor(num_channels / 2);

        let theta = tf.layers.conv2d({filters:num_channels_attn, kernelSize:(1, 1), useBias:false,
                                      strides:(1, 1),
                                      padding:'same',
                                      kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                      kernelRegularizer:tf.regularizers.l2(),
                                      name:"conv2d_theta"}).apply(input);
        theta = this._spatial_flatten(theta);

        let phi = tf.layers.conv2d({filters:num_channels_attn, kernelSize:(1, 1), useBias:false,
                                    strides:(1, 1),
                                    padding:'same',
                                    kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                    kernelIegularizer:tf.regularizers.l2(),
                                    name:"conv2d_phi"}).apply(input);
        phi = tf.layers.maxPooling2d({poolSize:[2, 2], strides:2}).apply(phi);
        phi = this._spatial_flatten(phi);

        let attn = theta.matMul(phi, false, true);
        attn = attn.softmax();

        let g = tf.layers.conv2d({filters:num_channels_g, kernelSize:(1, 1), useBias:false,
                                  strides:(1, 1),
                                  padding:'same',
                                  kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                  kernelRegularizer:tf.regularizers.l2(),
                                  name:"conv2d_g"}).apply(input);
        g = tf.layers.maxPooling2d({poolSize:[2, 2], strides:2}).apply(g);
        g = this._spatial_flatten(g);

        let attn_g = attn.matMul(g);
        attn_g = attn_g.reshape([attn_g.shape[0], h, -1, num_channels_g]);
        attn_g = tf.layers.conv2d({filters:num_channels, kernelSize:(1, 1), useBias:false, strides:(1, 1),
                                padding:'same',
                                   kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                kernelRegularizer:tf.regularizers.l2(),
                                name:"conv2d_attn_g"}).apply(attn_g);

        let result = this.sigma.val.mul(attn_g).add(input);
        return result;
    }

    computeOutputShape(inputShape){
        return inputShape;
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
        gat = tf.expandDims(gat, 0);
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
        this.c = config.c;
        this.B_list = [];
    }

    // call(z, y, num_blocks, vocab_size, embed_y, kwargs){
    call(inputs, kwargs){

        this.spatial_embedding = new SpatialEmbedding({vocab_size:this.vocab_size, filter_dim:this.embed_y});
        let z = inputs[0];
        let y = inputs[1].asType('int32');
        // let se_layer = spatial_embedding.apply(this.y);
        let se_layer = this.spatial_embedding.apply(y);

        let z_per_block = tf.split(z, this.num_blocks + 1, 1);
        let z0 = z_per_block[0];
        // z0 = tf.reshape(z0, (-1, 1, 1, z0.shape[1]));
        z0 = z0.reshape([-1, 1, 1, z0.shape[1]]);

        // let net = tf.linalg.matmul(tf.tile(z0, [1, tf.shape(se_layer)[1], 1, 1]), se_layer);
        let net = tf.tile(z0, [1, se_layer.shape[1], 1, 1]);
        net = net.matMul(se_layer);
        // net = se_layer.matMul(net);
        net = tf.squeeze(net, 2);

        net = net.reshape([net.shape[0], 512, this.seed_size_h, this.seed_size_w, -1]);
        net = net.reshape([net.shape[0], -1, 512, this.seed_size_h]);
        net = net.transpose([0, 3, 1, 2]);

        for (let block_idx = 0; block_idx < this.num_blocks; block_idx++){
            let name = "B"+(block_idx + 1).toString();
            let is_last_block = block_idx == this.num_blocks - 1;
            let res = new ResNetBlockUp({name:name, output_dim:this.out_channels[block_idx],
                                     is_last_block:is_last_block,
                                     conditioning_vector:z_per_block[block_idx],
                                     k_reg:this.kernel_reg});
            this.B_list.push(res);
            net = res.apply(net);
            if (this.blocks_with_attention.includes(name)){
                let nlb = new NonLocalBlock({name:name, k_reg:this.kernel_reg});
                this.B_list.push(nlb);
                net = nlb.apply(net);
            }
        }
        this.bn = tf.layers.batchNormalization();
        net = this.bn.apply(net);
        net = tf.relu(net);
        this.conv2d = tf.layers.conv2d({filters:this.c, kernelSize:(3, 3),
                                        strides:(1, 1),
                                        kernelInitializer:tf.initializers.randomNormal({mean:0.0, stddev:1.0}),
                                        kernelRegularizer:tf.regularizers.l2(),
                                        padding:'same'});
        net = this.conv2d.apply(net);

        net = tf.tanh(net);
        return net;
    }

    computeOutputShape(inputShape){
        return [1, 16, 16*inputShape[1][0], 1];
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
                                     c:c,
                                     embed_y:embed_y});
    let net = genPrep.apply([z, y]);
    // let z_per_block = genPrep.z_per_block;


    // model = tf.keras.Model([z, y], net);
    let model = tf.model({inputs:[z, y], outputs:net});

    return model;
}
function eval(seed, labels, g){
    let res = g.apply([seed, labels]);
    // let res = g.predict([seed, labels]);
    console.log("Result!");
    res.print();
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    // res = tf.expandDims(res, 2);
    const printCanvas = document.getElementById("lala");
    console.log(printCanvas);
    const image = tf.browser.toPixels(res, printCanvas);
    console.log(res);
    res.print();
}

let seed = tf.randomNormal([1, 128]);
let labels = tf.tensor([0, 1, 2], undefined, 'int32');
let g = makeGenerator(128, [32, 160, 1], [32, 8192], '', 'spectral_norm', 'B3', 52, false);
let sp_emb_kernel = null;

let toload = [
    "SpatialEmbedding_1-filter_bank.npy",
    "CBN_B1_1_dense_1-kernel.npy",
    "CBN_B1_1_batchNormalization-moving_mean.npy",
    "CBN_B1_1_batchNormalization-moving_variance.npy",
    "CBN_B1_1_dense_2-kernel.npy", // 4
    "B1_conv2dTranspose_1-kernel.npy",
    "B1_conv2dTranspose_1-bias.npy", // 6
    "CBN_B1_2_dense_1-kernel.npy",
    "CBN_B1_2_batchNormalization-moving_mean.npy",
    "CBN_B1_2_batchNormalization-moving_variance.npy",
    "CBN_B1_2_dense_2-kernel.npy", // 10
    "B1_conv2d-kernel.npy",
    "B1_conv2d-bias.npy", // 12
    "B1_conv2dTranspose_2-kernel.npy",
    "B1_conv2dTranspose_2-bias.npy", // 14
    "CBN_B2_1_dense_1-kernel.npy",
    "CBN_B2_1_batchNormalization-moving_mean.npy",
    "CBN_B2_1_batchNormalization-moving_variance.npy",
    "CBN_B2_1_dense_2-kernel.npy",
    "B2_conv2dTranspose_1-kernel.npy",
    "B2_conv2dTranspose_1-bias.npy",
    "CBN_B2_2_dense_1-kernel.npy",
    "CBN_B2_2_batchNormalization-moving_mean.npy",
    "CBN_B2_2_batchNormalization-moving_variance.npy",
    "CBN_B2_2_dense_2-kernel.npy",
    "B2_conv2d-kernel.npy",
    "B2_conv2d-bias.npy",
    "B2_conv2dTranspose_2-kernel.npy",
    "B2_conv2dTranspose_2-bias.npy", // 28
    "CBN_B3_1_dense_1-kernel.npy",
    "CBN_B3_1_batchNormalization-moving_mean.npy",
    "CBN_B3_1_batchNormalization-moving_variance.npy",
    "CBN_B3_1_dense_2-kernel.npy",
    "B3_conv2dTranspose_1-kernel.npy",
    "B3_conv2dTranspose_1-bias.npy",
    "CBN_B3_2_dense_1-kernel.npy",
    "CBN_B3_2_batchNormalization-moving_mean.npy",
    "CBN_B3_2_batchNormalization-moving_variance.npy",
    "CBN_B3_2_dense_2-kernel.npy",
    "B3_conv2d-kernel.npy",
    "B3_conv2d-bias.npy",
    "B3_conv2dTranspose_2-kernel.npy",
    "B3_conv2dTranspose_2-bias.npy", // 42
    "NonLocalBlock_B3_1-NonLocalBlock_B3_sigma.npy",
    "generatorPrep_batchNormalization-gamma.npy",
    "generatorPrep_batchNormalization-beta.npy",
    "generatorPrep_batchNormalization-moving_mean.npy",
    "generatorPrep_batchNormalization-moving_variance.npy",
    "generatorPrep_cond2d-kernel.npy",
    "generatorPrep_cond2d-bias.npy"];
let loadednpy = {};
let promises = [];

toload.map((fn) => {
    promises.push(n.load(fn).then(res => {
        let t = tf.tensor(res.data, res.shape);
        return t;
    }));
});


Promise.all(promises).then((v) => {
    console.log("done promise!");
    console.log(v);
    eval(seed, labels, g);
    let genPrep = g.layers[2];
    let B1 = genPrep.B_list[0];
    let B2 = genPrep.B_list[1];
    let B3 = genPrep.B_list[2];
    let NLB = genPrep.B_list[3];

    genPrep.spatial_embedding.setWeights([v[0]]);

    B1.cbn.dense1.setWeights([v[1]]);
    B1.cbn.bn.setWeights([v[2], v[3]]);
    B1.cbn.dense2.setWeights([v[4]]);
    B1.conv2dT1.setWeights([v[5], v[6]]);
    B1.cbn2.dense1.setWeights([v[7]]);
    B1.cbn2.bn.setWeights([v[8], v[9]]);
    B1.cbn2.dense2.setWeights([v[10]]);
    B1.conv2d.setWeights([v[11], v[12]]);
    B1.conv2dT2.setWeights([v[13], v[14]]);

    let s = 14;
    B2.cbn.dense1.setWeights([v[s+1]]);
    B2.cbn.bn.setWeights([v[s+2], v[s+3]]);
    B2.cbn.dense2.setWeights([v[s+4]]);
    B2.conv2dT1.setWeights([v[s+5], v[s+6]]);
    B2.cbn2.dense1.setWeights([v[s+7]]);
    B2.cbn2.bn.setWeights([v[s+8], v[s+9]]);
    B2.cbn2.dense2.setWeights([v[s+10]]);
    B2.conv2d.setWeights([v[s+11], v[s+12]]);
    B2.conv2dT2.setWeights([v[s+13], v[s+14]]);

    s = 28;
    B3.cbn.dense1.setWeights([v[s+1]]);
    B3.cbn.bn.setWeights([v[s+2], v[s+3]]);
    B3.cbn.dense2.setWeights([v[s+4]]);
    B3.conv2dT1.setWeights([v[s+5], v[s+6]]);
    B3.cbn2.dense1.setWeights([v[s+7]]);
    B3.cbn2.bn.setWeights([v[s+8], v[s+9]]);
    B3.cbn2.dense2.setWeights([v[s+10]]);
    B3.conv2d.setWeights([v[s+11], v[s+12]]);
    B3.conv2dT2.setWeights([v[s+13], v[s+14]]);

    s = 42;
    NLB.setWeights([v[s+1]]);
    genPrep.bn.setWeights([v[s+2], v[s+3], v[s+4], v[s+5]]);
    genPrep.conv2d.setWeights([v[s+6], v[s+7]]);
    eval(seed, labels, g);
});


// let npyLoaded = $.when(n);

function loadnpy(li){
    console.log(g.getWeights());
    g.summary();
    eval(seed, labels, g);
};



