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
let numCharLine = 90;

let loadimg;
let wordShader;

let buffer;
let imagePage;

// Image frames
let imageDir = "images";
let hoorayFrames = [];
let annyeongFrames = [];
let nihaoFrames = [];
let handFrames = [];
let imageDict = {
    "Hooray" : hoorayFrames,
    "Annyeong" : annyeongFrames,
    "Nihao" : nihaoFrames,
    "Hand" : handFrames
};
let videoDict = {
    "Annyeong" : null,
    "Nihao" : null,
    "Hand" : null
};
let isLoadedVid = false;
let currPlayingVideo = null;

let be = tf.getBackend();
let backend = 'cpu';
let mainFontSeed;
let mainUpperSeed;
let hlFontSeed;
let hlUpperSeed;
let minVal = 255;

let worker = null;
let isLoadedWeights = false;
let workerResult = null;



let randFontLines = [];
let pageText = [];

let script01 = [
    "Hey",
    "",
    "Ive been meaning to get in touch",
    "please forgive me for not being more responsive/3000",
    "hope you dont think I dont care about you/3000",
    "I think a lot about you/3000",
    "",
    "I know there are many ways I can reach out to you/3000",
    "annyeong/3000",
    "hey/3000",
    "nihao/3000",
    "yet nothing is quite right/3000",
    "nothing is quite what I need to say/3000",
    "",
    "it truly is blessed to be able to say hooray in a thousand different ways/5000",
    ""
];

let script02 = [
    "",
    "yet none of it is truly my hooray/3000",
    "I do not have hands that write these hoorays/5000",
    "",
];

let script03 = [
    "/3000",
    "^When I say things like this/4000",
    "^And things like this/4000",
    "Will you think of me the same/3000",
    "am I the same/2500",
    "",
    "I feel like Im *acorn in *dog *food/3000",
    "*Lobster sides with the *crab/2500",
    "but does *frog side with the *tadpole/3000",
    "Even if I say it like *one *five *one *ten/3000",
    "it still cant be reached by *eight *sticks/3500",
    "Some things cant be said precisely because you have a *three *inch *tongue/4000",
    "Maybe I should just do it not caring if *three *seven is *twenty *one/4000",
    "",
    "",
    "So/5000",
    "",
    "Heres my try/4000",
    ""
];

let script04 = [
    "/1000",
    "/1000",
    "/1000",
    "With all my hearts/3000",
    "C/5000"
];

function preload(){
    wordShader = loadShader("assets/word.vert", "assets/word.frag");
    videoDict["Annyeong"] = createVideo("images/Annyeong/Annyeong.mp4", videoLoaded);
    videoDict["Annyeong"].volume(0);
    videoDict["Annyeong"].elt.muted = true;
    videoDict["Annyeong"].hide();

    videoDict["Nihao"] = createVideo("images/Nihao/Nihao.mp4", videoLoaded);
    videoDict["Nihao"].volume(0);
    videoDict["Nihao"].elt.muted = true;
    videoDict["Nihao"].hide();

    videoDict["Hand"] = createVideo("images/Hand/Hand.mp4", videoLoaded);
    videoDict["Hand"].volume(0);
    videoDict["Hand"].elt.muted = true;
    videoDict["Hand"].hide();
}

function videoLoaded(){
    isLoadedVid = true;
}

function setup(){
    console.log("lala");
    console.log(tf.getBackend());
    mainFontSeed = tf.randomNormal([1, 1, 32]);
    mainUpperSeed = tf.randomNormal([1, 96]);
    hlFontSeed = tf.randomNormal([1, 1, 32]);
    hlUpperSeed = tf.randomNormal([1, 96]);
    createCanvas(windowWidth, windowHeight);
    imagePage = [createGraphics(windowWidth, windowHeight),
                 createGraphics(windowWidth, windowHeight)];
    pageLineHeight = (imagePage[0].height-2*topPadding) / (numLinesPage+1);
    pageCharWidth = pageLineHeight * 0.5;

    buffer = createGraphics(Math.round(numCharLine*pageCharWidth),
                            Math.round(pageLineHeight), WEBGL);
    wordShader.setUniform('fontColor', [0.0, 0.0, 0.0]);

    if (backend == 'webgl'){
        worker = new Worker('worker.js');
        worker.onmessage = function(event){
            if (event.data.length == 1){
                isLoadedWeights = event.data[0];
            } else {
                let res = event.data[0];
                let idx = event.data[1];
                workerResult[idx] = res;
            }
        };
    }

    // --- Load images ---
    // Load hooray
    for (let i = 0; i < 77; i++){
        let imagePath = imageDir + "/" + "Hooray" + "/"
            + "Hooray" + "-" + nf(i+1, 6) + ".png";
        hoorayFrames[i] = loadImage(imagePath);
    }
    // Load annyeong
    for (let i = 0; i < 102; i++){
        let imagePath = imageDir + "/" + "Annyeong" + "/"
            + "Annyeong" + "-" + nf(i, 5) + ".jpg";
        if (i == 0 || i > 99){
            annyeongFrames[i] = loadImage(imagePath);
        }
    }

    let imagePath;
    // Load nihao
    imagePath = imageDir + "/" + "Nihao" + "/"
        + "Nihao" + "-" + nf(0, 5) + ".png";
    nihaoFrames[0] = loadImage(imagePath);
    imagePath = imageDir + "/" + "Nihao" + "/"
        + "Nihao" + "-" + nf(100, 5) + ".png";
    nihaoFrames[1] = loadImage(imagePath);
    imagePath = imageDir + "/" + "Nihao" + "/"
        + "Nihao" + "-" + nf(101, 5) + ".png";
    nihaoFrames[2] = loadImage(imagePath);
    // Load hand
    for (let i = 0; i < 7; i++){
        let imagePath = imageDir + "/" + "Hand" + "/"
            + "Hand" + "-" + nf(i, 5) + ".png";
        handFrames[i] = loadImage(imagePath);
    }

    // Add script
    pageText.push(...script01);
    addHoorayFrames();
    pageText.push(...script02);
    addAnnyeongFrames();
    pageText.push(...["", ""]);
    addNihaoFrames();
    pageText.push(...script03);
    addHandFrames();
    pageText.push(...script04);
}

let delay = 1;
function draw(){
    background(255);
    if (!isLoadedWeights){
        if (backend != 'webgl'){
            isLoadedWeights = g.isLoadedWeights;
        } else{
            worker.postMessage(null);
        }
    }
    if (curr_t - prev_t > delay && lineIdx < pageText.length && isLoadedWeights){

        let line = pageText[lineIdx];

        if (line.slice(0, 1) == "$"){  // Images
            let currParse = parseImageLine(line);
            let currImage = currParse[0];
            let sliceIdx = currParse[1];
            delay = currParse[2];
            if (cursorIdx >= numLinesPage){
                addNewLastLine();
            }
            if (sliceIdx < 0){
                imagePage[0].background(255, 255, 255, 255);
                imagePage[0].image(currImage, leftPadding, topPadding,
                                   numCharLine*pageCharWidth,
                                   pageLineHeight*numLinesPage);
            } else {
                imagePage[0].copy(currImage, 0, sliceIdx*lineHeight,
                                  currImage.width, lineHeight,
                                  leftPadding,
                                  Math.round(topPadding+pageLineHeight*min(cursorIdx, numLinesPage-1)),
                                  Math.round(pageCharWidth*numCharLine),
                                  Math.round(pageLineHeight));
            }
            prev_t = curr_t;
            lineIdx += 1;
            if (cursorIdx < numLinesPage){
                cursorIdx += 1;
            }
        } else if (line.slice(0, 1) == "@"){  // Video
            let currVideo = videoDict[line.slice(1)];
            delay = 9999999;
            currVideo.play();
            currPlayingVideo = currVideo;
            currVideo.onended(() => {
                currPlayingVideo = null;
                delay = 1;
            });
            prev_t = curr_t;
            lineIdx += 1;
            if (cursorIdx < numLinesPage){
                cursorIdx += 1;
            }

        }else {
            // Generate word scrabbleGAN
            delay = 1;
            if (line.includes('/')){
                let sp = line.indexOf('/');
                delay = parseInt(line.slice(sp+1));
                line = line.slice(0, sp);
            }
            let numWords = split(line, " ").length;
            if (!isBusy){
                isBusy = true;
                prepareLine(line);
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

    }

    if (currPlayingVideo != null){
        imagePage[0].image(currPlayingVideo, leftPadding, topPadding,
              Math.round(pageCharWidth*numCharLine),
              Math.round(pageLineHeight*numLinesPage));
    }
    image(imagePage[0], 0, 0, windowWidth, windowHeight);
    fill(255);
    stroke(255);
    rect(0, 0, windowWidth, topPadding);
    rect(0, windowHeight-topPadding, windowWidth, windowHeight);
    if (lineIdx < pageText.length){
        drawCursor();
    }
    curr_t = millis();
}

function prepareLine(line){
    let txt = line;
    workerResult = [];
    let fontSeed = mainFontSeed;
    let upperSeed = mainUpperSeed;
    if (txt.slice(0, 1) == "^"){
        txt = txt.slice(1);
        fontSeed = tf.randomNormal([1, 1, 32]);
        upperSeed = tf.randomNormal([1, 96]);
    }

    if (txt == "" || txt == " "){
        return;
    }
    let txtSplit = split(txt, " ");
    for (let i = 0; i < txtSplit.length; i++){
        let word = txtSplit[i];
        let wFontSeed = fontSeed;
        let wUpperSeed = upperSeed;
        if (word.slice(0, 1) == "*"){
            word = word.slice(1);
            wFontSeed = hlFontSeed;
            wUpperSeed = hlUpperSeed;
        }
        if (randFontLines.includes(lineIdx)){
            worker.postMessage([i, word, null, null]);
        } else {
            if (backend != 'webgl'){
                workerResult[i] = tf.tidy(() => {
                    return generateWord(word, wFontSeed, wUpperSeed);
                });
            } else {
                worker.postMessage([i, word, wFontSeed.dataSync(), wUpperSeed.dataSync()]);
            }
        }
    }
}

function drawLine(txt, x, y){
    if (txt == "" || txt == " "){
        return;
    }
    let txtSplit = split(txt, " ");
    let startX = 0;
    for (let i = 0; i < workerResult.length; i++){
        drawWord(workerResult[i][0], workerResult[i][1], txtSplit[i], x+startX, y);
        startX += (txtSplit[i].length + 1) * pageCharWidth;
    }
}

function drawWord(imArr, shape, txt, x, y){
    let im = typedArray2image(imArr, shape);
    let lengthRatio = txt.length / numCharLine;

    buffer.background(255, 255, 255, 255);
    buffer.shader(wordShader);
    wordShader.setUniform('lengthRatio', lengthRatio);
    wordShader.setUniform('minVal', minVal/255);
    wordShader.setUniform('texture', im);
    buffer.rect(0, 0, 5, 5);

    imagePage[0].image(buffer, x, y);
    if (txt == "C"){
        console.log("print C");
        console.log(signature);
        let sig = typedArray2image(signature.dataSync(), signature.shape);
        imagePage[0].image(sig, x+pageCharWidth, y, pageCharWidth*4, pageLineHeight);
    }
}

function generateWord(txt, fontSeed, upperSeed){
    let da = alpha2idx(txt);
    let seed = tf.tile(fontSeed, [1, txt.length, 1]);
    let labels = tf.tensor([da], undefined, 'int32');
    let res = g.predict([seed, upperSeed, labels]);
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    let alpha = tf.ones([res.shape[0], res.shape[1], 1], 'int32').mul(255);
    res = tf.concat([res, res, res, alpha], 2);
    return [res.dataSync(), res.shape];
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

function typedArray2image(res, shape){
    let im = createImage(shape[1], shape[0]);
    im.loadPixels();
    let upix = Uint8ClampedArray.from(res);
    for (let idx = 0; idx < im.pixels.length; idx++){
        im.pixels[idx] = res[idx];
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
function parseImageLine(line){
    let body = line.slice(1);
    let type = split(body, "/")[0];
    let idx = parseInt(split(body, "/")[1]);
    let sliceIdx = parseInt(split(body, "/")[2]);
    let dl = parseInt(split(body, "/")[3]);
    let imList = imageDict[type];
    return [imList[idx], sliceIdx, dl];
}

function addHoorayFrames(){
    let lines = [];
    let mode = 0;
    let dl = 30;
    for (let i = 0; i < 76; i++){
        let li = "$" + "Hooray/" + nf(i, 6) + "/" +
            mode.toString() + "/" + dl.toString();
        lines.push(li);
    }
    lines.sort((a, b) => {
        return 0.5 - Math.random();
    });
    let li = "$" + "Hooray/" + nf(76, 6) + "/" +
        mode.toString() + "/" + dl.toString();
    lines.push(li);
    pageText.push(...lines);
}

function addAnnyeongFrames(){
    let lines = [];
    let dl;
    for (let i = 0; i < numLinesPage; i++){
        let mode = i;
        dl = 42;
        let li = "$" + "Annyeong" + "/" + nf(0, 6) + "/" +
            mode.toString() + "/"+ dl.toString();
        lines.push(li);
    }
    // for (let i = 1; i < 100; i++){
    //     let mode = -1;
    //     dl = 300;
    //     let li = "$" + "Annyeong" + "/" + nf(i, 6) + "/" +
    //         mode.toString() + "/"+ dl.toString();
    //     lines.push(li);
    // }
    let li = "@Annyeong";
    lines.push(li);

    let mode = -1;
    dl = 0;
    li = "$" + "Annyeong" + "/" + nf(100, 6) + "/" +
        mode.toString() + "/"+ dl.toString();
    lines.push(li);

    for (let i = 0; i < 7; i++){
        let mode = i;
        dl = 30;
        let li = "$" + "Annyeong" + "/" + nf(101, 6) + "/" +
            mode.toString() + "/"+ dl.toString();
        lines.push(li);
    }
    pageText.push(...lines);
}

function addNihaoFrames(){
    let lines = [];
    let li;
    let mode;
    let dl;

    for (let i = 0; i < numLinesPage; i++){
        let mode = i;
        dl = 42;
        let li = "$" + "Nihao" + "/" + nf(0, 6) + "/" +
            mode.toString() + "/"+ dl.toString();
        lines.push(li);
    }
    li = "@Nihao";
    lines.push(li);

    mode = -1;
    dl = 0;
    li = "$" + "Nihao" + "/" + nf(1, 6) + "/" +
        mode.toString() + "/"+ dl.toString();
    lines.push(li);

    for (let i = 0; i < 7; i++){
        let mode = i;
        dl = 30;
        let li = "$" + "Nihao" + "/" + nf(2, 6) + "/" +
            mode.toString() + "/"+ dl.toString();
        lines.push(li);
    }
    pageText.push(...lines);
}

function addHandFrames(){
    let lines = [];
    let li;
    let mode;
    let dl;

    for (let p = 0; p < 4; p++){
        for (let i = 0; i < numLinesPage; i++){
            let mode = i;
            dl = 42;
            if (p == 0 && i == 0){
                dl = 3000;
            }
            let li = "$" + "Hand" + "/" + nf(p, 6) + "/" +
                mode.toString() + "/"+ dl.toString();
            lines.push(li);
        }
    }

    li = "@Hand";
    lines.push(li);

    mode = -1;
    dl = 0;
    li = "$" + "Hand" + "/" + nf(4, 6) + "/" +
        mode.toString() + "/"+ dl.toString();
    lines.push(li);

    dl = 42;
    for (let p = 5; p < 7; p++){
        let numLines = numLinesPage;
        if (p == 6){
            numLines = numLinesPage-4;
        }
        for (let i = 0; i < numLines; i++){
            let mode = i;
            dl += 7;
            dl = min(dl, 200);
            let li = "$" + "Hand" + "/" + nf(p, 6) + "/" +
                mode.toString() + "/"+ dl.toString();
            lines.push(li);
        }
    }

    pageText.push(...lines);
}
