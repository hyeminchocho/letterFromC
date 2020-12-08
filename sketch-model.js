function setup(){
    createCanvas(200, 200);
    let seed = tf.randomNormal([1, 128]);
    let da = alpha2idx("lalalalal");
    let labels = tf.tensor([da], undefined, 'int32');
    let res = g.predict([seed, labels]);
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    console.log("p5!!");
    console.log(res.dataSync());
    console.log(res.shape);
}

function draw(){
    background(0);
}
