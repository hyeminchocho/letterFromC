self.importScripts('tf.min.js');
self.importScripts('npy.js');
self.importScripts('layer.js');

self.onmessage = function(event) {
    if (event.data == null){
        self.postMessage([g.isLoadedWeights]);
        return;
    }
    let idx = event.data[0];
    let txt = event.data[1];
    let fontSeed = event.data[2];
    if (fontSeed == null){
        console.log("inside IFFF");
        fontSeed = tf.randomNormal([1, 1, 32]);
    } else {
        fontSeed = tf.tensor(fontSeed);
        fontSeed = tf.reshape(fontSeed, [1, 1, 32]);
    }
    console.log(fontSeed);
    let upperSeed = event.data[3];
    if (upperSeed == null){
        upperSeed = tf.randomNormal([1, 96]);
    } else {
        upperSeed = tf.tensor(upperSeed);
        upperSeed = tf.reshape(upperSeed, [1, 96]);
    }


    let da = alpha2idx(txt);
    let seed = tf.tile(fontSeed, [1, txt.length, 1]);
    let labels = tf.tensor([da], undefined, 'int32');
    let res = g.predict([seed, upperSeed, labels]);
    res = res.add(1.0).div(2.0).mul(255);
    res = res.squeeze([0]).asType('int32');
    let alpha = tf.ones([res.shape[0], res.shape[1], 1], 'int32').mul(255);
    res = tf.concat([res, res, res, alpha], 2);
    self.postMessage([[res.dataSync(), res.shape], idx]);
};
