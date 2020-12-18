let prev = 0;
let curr = 0;
let count = 32;
let currCount = 0;

let lineHeight = 32;
let charWidth = 16;
let pageLineHeight;
let pageCharWidth;

let numLinesPage = 37;
let numCharLine;

let loadimg;
let wordShader;

let buffer;
let imagePage;

function preload(){
    wordShader = loadShader("assets/word.vert", "assets/word.frag");
}

function setup(){
    createCanvas(windowWidth, windowHeight);
    imagePage = createGraphics(2412, 3074);

    pageLineHeight = imagePage.height / numLinesPage;
    pageCharWidth = pageLineHeight * 0.5;
    numCharLine = Math.ceil(imagePage.width/pageCharWidth);
    console.log(imagePage.width);
    console.log(pageCharWidth);
    console.log(numCharLine);

    buffer = createGraphics(numCharLine*pageCharWidth, pageLineHeight, WEBGL);
    wordShader.setUniform('fontColor', [0.0, 0.0, 0.0]);
}

let c = 0;
function draw(){
    if (curr - prev > 1 && currCount < count && g.isLoadedWeights){
        background(255);

        // drawWord("ippuniDASULDA", 5, 5);
        // drawWord("dasuldaIPPUNI", 5, 5+pageLineHeight);
        drawWord("distinctttttttttttttttttt", 5, 5+pageLineHeight*c);
        c+=1;

        prev = curr;
        currCount += 1;

        image(imagePage, 0, 0, width, height);
    }
    curr = millis();
}

function drawWord(txt, x, y){
    let im = generateWord(txt);
    let lengthRatio = txt.length / numCharLine;
    console.log(lengthRatio);

    buffer.background(255);
    buffer.shader(wordShader);
    wordShader.setUniform('lengthRatio', lengthRatio);
    wordShader.setUniform('texture', im);
    buffer.rect(0, 0, 5, 5);

    imagePage.image(buffer, x, y);
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
