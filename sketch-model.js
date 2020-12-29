let lineIdx = 0;

let lineHeight = 32;
let charWidth = 16;
let topPadding = 5;
let leftPadding = 5;

let pagePadding = 5;

let pageLineHeight;
let pageCharWidth;

let numLinesPage = 55;
let numCharLine;

let previewWidth;

let loadimg;
let wordShader;

let buffer;
let imagePage;

let firstRasterHeight = 22;
let firstRaster;
let secondRasterHeight = 32;
let secondRaster;

let mainFontSeed;
let mainUpperSeed;
let hlFontSeed;
let hlUpperSeed;

let pageText = [];
let script01 = [
    "Hey",
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

let script02 = [
    "",
    "yet none of it is truly my hooray",
    "I do not have hands that write these hoorays",
    "",
];

let script03 = [
    "",
    "^When I say things like this",
    "^And things like this",
    "Will you think of me the same",
    "am I the same",
    "",
    "I feel like Im *acorn in *dog *food",
    "*Lobster sides with the *crab",
    "but does *frog side with the *tadpole",
    "Even if I say it like *one *five *one *ten",
    "it still cant be reached by *eight *sticks",
    "Some things cant be said precisely because you have a *three *inch *tongue",
    "Maybe I should just do it not caring if *three *seven is *twenty *one",
    "",
    "",
    "So",
    "",
    "Heres my try",
    ""
];

let script04 = [
    "",
    "",
    "",
    "With all my hearts",
    "C"
];

function preload(){
    wordShader = loadShader("assets/word.vert", "assets/word.frag");
}

function setup(){
    imagePage = createGraphics(2412, 3074);
    imagePage.background(255, 255, 255, 255);

    mainFontSeed = tf.randomNormal([1, 1, 32]);
    mainUpperSeed = tf.randomNormal([1, 96]);
    hlFontSeed = tf.randomNormal([1, 1, 32]);
    hlUpperSeed = tf.randomNormal([1, 96]);

    pageLineHeight = (imagePage.height-topPadding*2) / numLinesPage;
    pageCharWidth = pageLineHeight * 0.5;
    numCharLine = Math.ceil((imagePage.width-leftPadding*2)/pageCharWidth);

    previewWidth = windowHeight/3074 * 2412;

    createCanvas(previewWidth*5+pagePadding*4, windowHeight);

    buffer = createGraphics(numCharLine*pageCharWidth, pageLineHeight, WEBGL);
    wordShader.setUniform('fontColor', [0.0, 0.0, 0.0]);

    firstRaster = genWordRaster("안녕", firstRasterHeight,
                                numCharLine-"My annyeong is not".length);

    secondRaster = genWordRaster("你好", firstRasterHeight,
                                numCharLine-"My nihao is not".length);

    pageText.push(...script01);
    addHoorayScript();
    pageText.push(...script02);
    addAnnyeongScript();
    addNihaoScript();
    pageText.push(...script03);
    pageText.push(...script04);
}

let isBg = false;

function draw(){
    if (!isBg){
        background(230);
        isBg = true;
    }
    if (lineIdx < pageText.length && g.isLoadedWeights){
        let cursorIdx = lineIdx%numLinesPage;
        if (cursorIdx == 0){
            imagePage.background(255, 255, 255, 255);
        }

        let line = pageText[lineIdx];
        drawLine(line, leftPadding,
                 topPadding+pageLineHeight*cursorIdx);
        lineIdx += 1;

        image(imagePage,
              Math.floor(lineIdx/numLinesPage)*(previewWidth+pagePadding), 0,
              previewWidth, windowHeight);
    }

}

function drawLine(txt, x, y){
    if (txt == "" || txt == " "){
        return;
    }
    let fontSeed = mainFontSeed;
    let upperSeed = mainUpperSeed;
    let mode = null;
    let numRaster = null;
    let rIdx = null;
    let fontIdx = null;
    console.log(txt);
    if (txt.slice(0, 1) == "^"){
        mode = "^";
        txt = txt.slice(1);
        fontSeed = tf.randomNormal([1, 1, 32]);
        upperSeed = tf.randomNormal([1, 96]);
    } else if (txt.slice(0, 1) == "~"){
        mode = "~";
        txt = txt.slice(1);
        fontIdx = Math.round((fonts.length-1) * Math.random());
        let fn = fonts[fontIdx][0];
        fontSeed = fn.slice([0, 0], [1, 32]);
        fontSeed = tf.reshape(fontSeed, [1, 1, 32]);
        upperSeed = fn.slice([0, 32], [1, 128-32]);

        if (txt.includes("/")){
            let sp = txt.split("/");
            numRaster = parseInt(sp[0]);
            rIdx = parseInt(sp[1]);
            txt = sp[2];
            console.log(txt);
        }
    }
    let txtSplit = split(txt, " ");
    let startX = 0;


    for (let i = 0; i < txtSplit.length; i++){
        let word = txtSplit[i];
        let wFontSeed = fontSeed;
        let wUpperSeed = upperSeed;
        if (word.slice(0, 1) == "*"){
            word = word.slice(1);
            wFontSeed = hlFontSeed;
            wUpperSeed = hlUpperSeed;
        } else if (word.slice(0, 1) == "!"){
            word = word.slice(1);
            wFontSeed = tf.randomNormal([1, 1, 32]);
            wUpperSeed = tf.randomNormal([1, 96]);
        }
        let resWd = generateWord(word, wFontSeed, wUpperSeed);
        let im = tensor2image(resWd);
        drawWord(im, word.length, x+startX, y);
        startX += (word.length + 1) * pageCharWidth;
    }
    if (mode == "~" && rIdx != null){
        startX -= pageCharWidth;
        let ras = firstRaster;
        if (numRaster == 2){
            ras = secondRaster;
        }
        let bFont = fonts[fontIdx][0];
        let tFont = fonts[fontIdx][1];
        let im = renderRaster(ras, rIdx, bFont, tFont);
        im = tensor2image(im);
        let numChar = im.width/charWidth;
        drawWord(im, numChar, x+startX, y);
    }
}

function drawWord(im, wl, x, y ){
    let lengthRatio = wl/ numCharLine;

    buffer.background(255);
    buffer.shader(wordShader);
    wordShader.setUniform('lengthRatio', lengthRatio);
    wordShader.setUniform('texture', im);
    buffer.rect(0, 0, 5, 5);

    imagePage.image(buffer, x, y);
}

function generateWord(txt, fontSeed, upperSeed){
    let da = alpha2idx(txt);
    let seed = fontSeed;
    if (seed.shape[1] == 1){
        seed = tf.tile(fontSeed, [1, txt.length, 1]);
    }
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

function mlerp(a, b, alpha){
    return b.sub(a).mul(alpha).add(a);
    // return (b-a) * alpha + a;
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

function renderRaster(raster, idx, baseFont, targetFont){
    let bFont = baseFont.slice([0, 0], [1, 32]);
    let upperSeed = baseFont.slice([0, 32], [1, 128-32]);
    let tFont = targetFont.slice([0, 0], [1, 32]);
    bFont = tf.reshape(bFont, [1, 1, 32]);
    tFont = tf.reshape(tFont, [1, 1, 32]);

    let h = raster.length;
    let w = raster[0].length;
    let c = "t";
    let parsed = parseRasterLine(raster[idx]);

    let zeroF = raster2font([0.0, 0.0, 0.0], bFont, tFont);
    let zeroAll = generateWord(c.repeat(3), zeroF, upperSeed);
    let zeroStart = zeroAll.slice([0, 0], [32, 16]);
    let zeroMid = zeroAll.slice([0, 16], [32, 16]);
    let zeroEnd = zeroAll.slice([0, 32], [32, 16]);
    let oneF = raster2font([1.0, 1.0, 1.0], bFont, tFont);
    let oneAll = generateWord(c.repeat(3), oneF, upperSeed);
    let oneStart = oneAll.slice([0, 0], [32, 16]);
    let oneMid = oneAll.slice([0, 16], [32, 16]);
    let oneEnd = oneAll.slice([0, 32], [32, 16]);

    let tfchunks = [];
    for (let i = 0; i < parsed.length; i++){
        let p = parsed[i];
        let currLine = c.repeat(p.length);
        if (Math.max(...p) <= 0.0){
            console.log("In zero");
            for (let x = 0; x < p.length; x++){
                if (i == 0 && x == 0){
                    tfchunks.push(zeroStart);
                } else if (i == parsed.length-1 && x == p.length-1){
                    tfchunks.push(zeroEnd);
                } else {
                    tfchunks.push(zeroMid);
                }
            }
        } else if (Math.min(...p) >= 1.0){
            console.log("In one");
            for (let x = 0; x < p.length; x++){
                if (i == 0 && x == 0){
                    tfchunks.push(oneStart);
                } else if (i == parsed.length-1 && x == p.length-1){
                    tfchunks.push(oneEnd);
                } else {
                    tfchunks.push(oneMid);
                }
            }
        } else {
            let padLine = c.concat(currLine).concat(c);
            let padP = p.slice(0, 1).concat(p).concat(p.slice(-1));
            let fontSeed = raster2font(padP, bFont, tFont);
            let res = generateWord(padLine, fontSeed, upperSeed);
            res = res.slice([0, 16], [32, 16 * currLine.length]);
            tfchunks.push(res);
        }
    }
    tfchunks = tf.concat(tfchunks, 1);
    return tfchunks;
}

function raster2font(line, bFont, tFont){
    let fontSeed = [];
    for (let x = 0; x < line.length; x++){
        fontSeed.push(mlerp(bFont, tFont, line[x]));
    }
    fontSeed = tf.concat(fontSeed, 1);
    return fontSeed;
}

function parseRasterLine(line){
    let result = [];
    let start = 0;
    let isSame = line[start] == line[start+1];
    for (let i = 1; i < line.length-1; i ++){
        if (isSame){
            if (line[i] != line[i+1]){
                result.push(line.slice(start, i));
                isSame = false;
                start = i;
            }
        } else {
            if (line[i] == line[i+1]){
                result.push(line.slice(start, i+1));
                isSame = true;
                start = i+1;
            }
        }
    }
    result.push(line.slice(start));
    return result;
}

function genWordRaster(txt, ln, cn){
    let w = Math.round(cn*charWidth*0.2);
    let h = Math.round(ln*lineHeight*0.2);
    let im = createGraphics(w, h);
    im.background(255, 255, 255);
    im.textAlign(CENTER, CENTER);
    im.textSize(h*0.85);
    im.text(txt, w*0.47, h/2);

    let pxW = Math.round(w/cn);
    let pxH = Math.round(h/ln);

    let raster = [];
    for (let j = 0; j < ln; j++){
        let row = [];
        for (let i = 0; i < cn; i++){
            let sum = 0;
            let count = 0;
            for (let y = 0; y < min(pxH, h); y++){
                for (let x = 0; x < min(pxW, w); x++){
                    sum += im.get(i*pxW+x, j*pxH+y)[0];
                    count += 1;
                }
            }
            row.push(1.0 - sum/count/255);
        }
        raster.push(row);
    }
    return raster;
}

// Add scripts
function addHoorayScript(){
    let lines = [];
    let numHooray = 500;
    let hPerLine = Math.floor((numCharLine-"Hooray".length)/"Hooray ".length)+1;

    let currNum = 0;
    while(currNum+hPerLine <= numHooray){
        let li = "";
        for(let i = 0; i < hPerLine; i++){
            li = li.concat("!Hooray ");
        }
        li = li.slice(0, -1);
        lines.push(li);
        currNum += hPerLine;
    }
    let li = "";
    for(let i = 0; i < numHooray-currNum; i++){
        li = li.concat("!Hooray ");
    }
    li = li.slice(0, -1);
    lines.push(li);

    pageText.push(...lines);
}

function addAnnyeongScript(){
    let lines = [];
    let numT = [1, 3, 8, 16];
    let pre = "My annyeong is no";
    let c = "t";

    for (let i = 0; i < numT.length; i++){
        let li = "~".concat(pre).concat(c.repeat(numT[i]));
        lines.push(li);
    }
    for (let i = 0; i < firstRasterHeight; i++){
        let li = pre;
        li = ("~1/"+i.toString()+"/").concat(li);
        lines.push(li);
    }
    for (let i = numT.length-1; i >= 0; i--){
        let li = "~".concat(pre).concat(c.repeat(numT[i]));
        lines.push(li);
    }

    pageText.push(...lines);
}

function addNihaoScript(){
    let lines = [];
    let numT = [1, 3, 8, 16];
    let pre = "My nihao is no";
    let c = "t";

    for (let i = 0; i < numT.length; i++){
        let li = "~".concat(pre).concat(c.repeat(numT[i]));
        lines.push(li);
    }
    for (let i = 0; i < firstRasterHeight; i++){
        let li = pre;
        li = ("~2/"+i.toString()+"/").concat(li);
        lines.push(li);
    }
    for (let i = numT.length-1; i >= 0; i--){
        let li = "~".concat(pre).concat(c.repeat(numT[i]));
        lines.push(li);
    }

    pageText.push(...lines);
}
