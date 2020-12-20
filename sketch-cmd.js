let count = 1;
let lineIdx = 0;

let lineHeight = 32;
let charWidth = 16;
let topPadding = 5;
let leftPadding = 5;

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
let isLoadedWeights = false;
let workerResult = null;
worker.onmessage = function(event){
    if (event.data.length == 1){
        isLoadedWeights = event.data[0];
    } else {
        let res = event.data[0];
        let idx = event.data[1];
        workerResult[idx] = res;
    }
};

let randFontLines = [];
let pageText = [
    "Dear hyemin",
    "",
    "Ive been meaning to get in touch",
    "please forgive me for not being more responsive",
    "hope you dont think I dont care about you",
    "I think a lot about you",
    "",
    "I know there are many ways I can reach out to you",
    "annyeong",
    "hey",
    "nihao",
    "yet nothing is quite right",
    "nothing is quite what I need to say",
    "",
    "it truly is blessed to be able to say hooray in a thousand different ways",
    ""
];

function preload(){
    wordShader = loadShader("assets/word.vert", "assets/word.frag");
}

function setup(){
    setAttributes('antialias', true);
    createCanvas(windowWidth, windowHeight);
    imagePage = [createGraphics(windowWidth, windowHeight),
                 createGraphics(windowWidth, windowHeight)];
    pageLineHeight = (imagePage[0].height-2*topPadding) / (numLinesPage+1);
    pageCharWidth = pageLineHeight * 0.5;
    numCharLine = Math.ceil(imagePage[0].width/pageCharWidth);

    buffer = createGraphics(numCharLine*pageCharWidth, pageLineHeight, WEBGL);
    wordShader.setUniform('fontColor', [0.0, 0.0, 0.0]);

    // Add script
    addHoorayScript();
    console.log(pageText);
}

function draw(){
    background(255);
    if (!isLoadedWeights){
        worker.postMessage(null);
    }
    if (curr_t - prev_t > 1 && lineIdx < pageText.length && isLoadedWeights){

        let line = pageText[lineIdx];
        let numWords = split(line, " ").length;
        if (!isBusy){
            isBusy = true;
            prepareLine(lineIdx);
        }
        if (workerResult.length >= numWords || line == ""){
            if (cursorIdx >= numLinesPage){
                addNewLastLine();
            }
            drawLine(line, leftPadding,
                     topPadding+pageLineHeight*min(cursorIdx, numLinesPage-1));

            prev_t = curr_t;
            lineIdx += 1;
            if (cursorIdx < numLinesPage){
                cursorIdx += 1;
            }
            isBusy = false;
        }

    }
    image(imagePage[0], 0, 0, windowWidth, windowHeight);
    drawCursor();
    curr_t = millis();
}

function prepareLine(lineIdx){
    let txt = pageText[lineIdx];
    workerResult = [];
    if (txt == "" || txt == " "){
        return;
    }
    let txtSplit = split(txt, " ");
    for (let i = 0; i < txtSplit.length; i++){
        if (randFontLines.includes(lineIdx)){
            worker.postMessage([i, txtSplit[i], null, null]);
        } else {
            worker.postMessage([i, txtSplit[i], fontSeed.dataSync(), upperSeed.dataSync()]);
        }
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

    buffer.background(255, 255, 255, 255);
    buffer.shader(wordShader);
    wordShader.setUniform('lengthRatio', lengthRatio);
    wordShader.setUniform('minVal', minVal/255);
    wordShader.setUniform('texture', im);
    buffer.rect(0, 0, 5, 5);

    imagePage[0].image(buffer, x, y);
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
        rect(leftPadding, topPadding+pageLineHeight*cursorIdx, pageCharWidth, pageLineHeight);
    }
}

function addNewLastLine(){
    let buf01 = imagePage[0];
    let buf02 = imagePage[1];
    buf02.background(255);
    buf02.image(buf01, 0, -pageLineHeight);
    imagePage = [buf02, buf01];
}

// -------- Coreo functions ---------
function addHoorayScript(){
    let numHooray = 1000;
    let hoorPerLine = Math.floor(numCharLine / "hooray ".length);
    let currNum = 0;
    let fullLine = "";
    for (let i = 0; i < hoorPerLine; i++){
        fullLine = fullLine.concat("hooray ");
    }
    fullLine = fullLine.slice(0, -1);
    while (currNum < numHooray){
        if ((currNum + hoorPerLine) < numHooray){
            pageText.push(fullLine);
            currNum += hoorPerLine;
        } else {
            let lastHoor = numHooray - currNum;
            let line = "";
            for (let i = 0; i < lastHoor; i++){
                line = line.concat("hooray ");
            }
            line = line.slice(0, -1);
            pageText.push(line);
            currNum += lastHoor;
        }
        randFontLines.push(pageText.length-1);
    }
}
