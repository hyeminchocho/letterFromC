let count = 1;
let currCount = 0;

let lineHeight = 32;
let charWidth = 16;
let pageLineHeight;
let pageCharWidth;

let cursorIdx = 0;
let curr_t = 0.0;
let timeUnit = 25;
let prev_t = -timeUnit*2;
let isBusy = false;

let numLinesPage = 32;
let numCharLine;

let loadimg;
let wordShader;

let buffer;
let imagePage;

let fontSeed = tf.randomNormal([1, 1, 32]);
let upperSeed = tf.randomNormal([1, 96]);
let minVal = 255;

let worker = new Worker('worker.js');
let workerResult = null;
worker.onmessage = function(event){
    let res = event.data[0];
    let idx = event.data[1];
    workerResult[idx] = res;
};

let pageText = [
    "Dear hyemin",
    "",
    "Ive been meaning to get in touch",
    "please forgive me for not being more responsive",
    "hope you dont think I dont care about you",
    "I think a lot about you",
    // "",
    // "I know there are many ways I can reach out to you",
    // "annyeong",
    // "hey",
    // "nihao",
    // "yet nothing is quite right",
    // "nothing is quite what I need to say",
    // "",
    // "it truly is blessed to be able to say hooray in a thousand different ways",
    // ""
];

function preload(){
    wordShader = loadShader("assets/word.vert", "assets/word.frag");
}

function setup(){
    createCanvas(windowWidth, windowHeight);
    imagePage = createGraphics(windowWidth, windowHeight);

    pageLineHeight = imagePage.height / numLinesPage;
    pageCharWidth = pageLineHeight * 0.5;
    numCharLine = Math.ceil(imagePage.width/pageCharWidth);

    buffer = createGraphics(numCharLine*pageCharWidth, pageLineHeight, WEBGL);
    wordShader.setUniform('fontColor', [0.0, 0.0, 0.0]);

}

function draw(){
    background(255);
    // if (curr_t - prev_t > 3000 && currCount < pageText.length && g.isLoadedWeights && !isBusy){
    if (curr_t - prev_t > 1 && currCount < pageText.length){

        let line = pageText[currCount];
        let numWords = split(line, " ").length;
        if (!isBusy){
            isBusy = true;
            prepareLine(line);
        }
        if (workerResult.length >= numWords || line == ""){
            drawLine(line, 5, 5+pageLineHeight*currCount);

            prev_t = curr_t;
            currCount += 1;
            isBusy = false;
        }

    }
    image(imagePage, 0, 0, windowWidth, windowHeight);
    drawCursor();
    curr_t = millis();
}

function prepareLine(txt){
    workerResult = [];
    if (txt == "" || txt == " "){
        return;
    }
    let txtSplit = split(txt, " ");
    for (let i = 0; i < txtSplit.length; i++){
        worker.postMessage([i, txtSplit[i], fontSeed.dataSync(), upperSeed.dataSync()]);
    }
}

function drawLine(txt, x, y){
    // isBusy = true;
    if (txt == "" || txt == " "){
        return;
    }
    let txtSplit = split(txt, " ");
    // workerResult = [];
    // for (let i = 0; i < txtSplit.length; i++){
    //     worker.postMessage([txt, fontSeed.dataSync(), upperSeed.dataSync()]);
    // }
    let startX = 0;
    for (let i = 0; i < workerResult.length; i++){
        drawWord(workerResult[i][0], workerResult[i][1], txtSplit[i], x+startX, y);
        startX += (txtSplit[i].length + 1) * pageCharWidth;
    }

    // isBusy = false;
}

function drawWord(imArr, shape, txt, x, y){
    // let imTensor = generateWord(txt);
    let im = typedArray2image(imArr, shape);
    let lengthRatio = txt.length / numCharLine;

    buffer.background(255);
    buffer.shader(wordShader);
    wordShader.setUniform('lengthRatio', lengthRatio);
    wordShader.setUniform('minVal', minVal/255);
    wordShader.setUniform('texture', im);
    buffer.rect(0, 0, 5, 5);

    imagePage.image(buffer, x, y);
}

function generateWord(txt){
    let da = alpha2idx(txt);
    let seed = tf.tile(fontSeed, [1, txt.length, 1]);
    // let a = tf.randomNormal([1, 1, 32]);
    // let b = tf.randomNormal([1, 1, 32]);
    // let firstS = tf.tile(a, [1, txt.length-8, 1]);
    // let seed = genLerpSeed(a, b, 8);
    // seed = tf.concat([firstS, seed], 1);
    let labels = tf.tensor([da], undefined, 'int32');
    let res = g.predict([seed, upperSeed, labels]);
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    let alpha = tf.ones([res.shape[0], res.shape[1], 1], 'int32').mul(255);
    res = tf.concat([res, res, res, alpha], 2);
    return res;
}

function genLerpSeed(a, b, length){
    let alphaArr = tf.range(0, 1, 1/length);
    alphaArr = tf.reshape(alphaArr, [1, length, 1]);
    alphaArr = tf.tile(alphaArr, [1, 1, 32]);

    let ra = tf.reshape(a, [1, 1, 32]);
    ra = tf.tile(a, [1, length, 1]);
    let rb = tf.reshape(b, [1, 1, 32]);
    rb = tf.tile(b, [1, length, 1]);
    return rb.sub(ra).mul(alphaArr).add(ra);
}

function lerp(a, b, alpha){
    return (b-a) * alpha + a;
}

function typedArray2image(res, shape){
    let im = createImage(shape[1], shape[0]);
    im.loadPixels();
    let upix = Uint8ClampedArray.from(res);
    for (let idx = 0; idx < im.pixels.length; idx++){
        im.pixels[idx] = upix[idx];
        if (upix[idx] < minVal){
            minVal = upix[idx];
        }
    }
    im.updatePixels();
    return im;
}

function tensor2image(res){
    let im = createImage(res.shape[1], res.shape[0]);
    im.loadPixels();
    let upix = Uint8ClampedArray.from(res.dataSync());
    for (let idx = 0; idx < im.pixels.length; idx++){
        im.pixels[idx] = upix[idx];
        if (upix[idx] < minVal){
            minVal = upix[idx];
        }
    }
    im.updatePixels();
    return im;
}

function drawCursor(){
    if (floor((curr_t-prev_t)/1500) % 2 == 0){
        fill(0, 0, 0);
        rect(10, 10+pageLineHeight*currCount, pageCharWidth, pageLineHeight);
    }
}
