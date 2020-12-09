let prev = 0;
let curr = 0;
let count = 120;
let currCount = 0;
let loadimg;

function preload(){
}

function setup(){
    createCanvas(200, 200);
}

function draw(){
    if (curr - prev > 1 && currCount < count && g.isLoadedWeights){
        background(155, 0, 155);
        console.log("p4");
        console.log(g.isLoadedWeights);

        let im = generateWord("kikikukuku");

        image(im, 0, 0);

        prev = curr;
        currCount += 1;
    }
    curr = millis();
}

function generateWord(txt){
    let da = alpha2idx(txt);
    let seed = tf.randomNormal([1, 128]);
    let labels = tf.tensor([da], undefined, 'int32');
    let res = g.predict([seed, labels]);
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    let alpha = tf.ones([res.shape[0], res.shape[1], 1], 'int32').mul(255);
    res = tf.concat([res, res, res, alpha], 2);
    return tensor2image(res);
}

function tensor2image(res){
    let im = createImage(res.shape[1], res.shape[0]);
    im.loadPixels();
    let upix = Uint8ClampedArray.from(res.dataSync());
    for (let idx = 0; idx < im.pixels.length; idx++){
        im.pixels[idx] = upix[idx];
    }
    im.updatePixels();
    return im;
}
