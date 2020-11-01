let inputDir = "generated/Animations";
let title = "Annyeong-201031-061051";
let numAnim = 17701;
let allImages = [];
let scrollPath = inputDir+"/"+title+"/"+title+"-scroll.json";
let scrollJSON;
let speedPath = inputDir+"/"+title+"/"+title+"-speed.json";
let speedJSON;
let currLines = [];
let curr_t = 0.0;

let linesPerPage = 36;
let scale = 0.5;
let charHeight = 32;
let lineHeight = charHeight * scale;
let timeUnit = 25;
let prev_t = -timeUnit*2;
let toLoadLineIdx = 0;
let scrollIdx = 0;

let scrollJSONLen = 6431;
let speedJSONLen = 6431;

let allLoaded = false;
let loadedNum = 0;
let isStatic = false;
let frame;

function preload(){
    frame = loadImage("generated/letterFromCframe.jpg");
}

function callback(loadedImg){
    loadedNum += 1;
}

function setup() {
    let url = getURL();
    if (url.endsWith('?frame')){
        isStatic = true;
        noLoop();
    }
    createCanvas(windowWidth, windowHeight, P2D);
    scale = ((windowHeight-20) / (linesPerPage+1)) / charHeight;
    lineHeight = charHeight*scale;
    background(255);

    if (!isStatic){
        scrollJSON = loadJSON(scrollPath);
        speedJSON = loadJSON(speedPath);
        for (let i = 0; i < numAnim; i++){
            let imagePath = inputDir + "/" + title + "/"
                + title + "-" + nf(i, 6) + ".jpg";
            allImages[i] = loadImage(imagePath, callback);
        }
    }
}

function windowResized(){
    resizeCanvas(windowWidth, windowHeight);
    scale = ((windowHeight-20) / (linesPerPage+1)) / charHeight;
    lineHeight = charHeight*scale;
}

function draw() {
    background(255);
    // image(allImages[3], 0, 0);
    if (!isStatic){
        if (allLoaded){
            prev_t = updateLines(curr_t, prev_t);
            let y = 0;
            for (let i = 0; i < currLines.length; i++){
                image(currLines[i], 10, y+10,
                      currLines[i].width*scale, currLines[i].height*scale);
                // console.log(currLines[i].height*scale);
                y += lineHeight;
                // console.log(scale);
                // console.log(lineHeight);
                // console.log(currLines[i].height);
            }
            drawCursor();
            curr_t = millis();
        } else {
            let perc = loadedNum / numAnim;
            stroke(0);
            strokeWeight(3);
            fill(255, 255, 255);
            rect(5, 5, 105 ,15);
            fill(0, 0, 0);
            rect(5, 5, 105*perc, 15);
            strokeWeight(1);
            text(nf(perc*100, 2, 2)+"%", 5, 35);
            // console.log(perc);
            if (perc > 0.99){
                allLoaded = true;
            }
        }
    } else {
        image(frame, 0, 0, frame.width*0.75, frame.height*0.75);
    }

}

function drawCursor(){
    let numLines = currLines.length;
    // if (numLines < linesPerPage){
    if (floor((curr_t-prev_t)/1500) % 2 == 0){
        fill(0, 0, 0);
        rect(10, 10+lineHeight*numLines, 16*scale, 32*scale);
    }
    // }
}

function updateLines(curr_t, prev_t){
    let diff_t = curr_t - prev_t;
    let should_diff = 60;
    if (scrollIdx < scrollJSONLen){
        should_diff = max(speedJSON[max(0, scrollIdx-1)], timeUnit);
    } else{
        should_diff = 9999999;
    }
    if (diff_t >= should_diff){
        let currScrollNum = 36;
        let currRemoveNum = 0;
        if (scrollIdx >= scrollJSONLen){
            currScrollNum = 0;
            currRemoveNum = 0;
        } else{
            currScroll = scrollJSON[max(0, scrollIdx-1)];
            currScrollNum = currScroll[0];
            currRemoveNum = currScroll[1];
        }
        for (let n = 0; n < currScrollNum; n++){
            if (toLoadLineIdx < numAnim){
                if (currLines.length > linesPerPage){
                    currLines.splice(0, 1);
                }

                currLines.push(allImages[toLoadLineIdx]);
                toLoadLineIdx += 1;
                toLoadLineIdx = min(toLoadLineIdx, endAnimIdx);
            }
        }
        // for (let n = 0; n < currRemoveNum; n++){
        //     if (currLines.length > 0){
        //         currLines.splice(0, 1);
        //     }
        // }
        scrollIdx += 1;
        return curr_t;
    }
    return prev_t;
}
