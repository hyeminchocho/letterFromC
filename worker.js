self.importScripts('tf.min.js');
self.importScripts('npy.js');
self.importScripts('layer.js');

self.onmessage = function(event) {
    if (event.data == null){
        self.postMessage([g.isLoadedWeights]);
        return;
    }
    // function generateWord(txt){
    let idx = event.data[0];
    let txt = event.data[1];
    let fontSeed = tf.tensor(event.data[2]);
    fontSeed = tf.reshape(fontSeed, [1, 1, 32]);
    let upperSeed = tf.tensor(event.data[3]);
    upperSeed = tf.reshape(upperSeed, [1, 96]);
    // let fontSeed = tf.randomNormal([1, 1, 32]);
    // let upperSeed = tf.randomNormal([1, 96]);
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
    self.postMessage([[res.dataSync(), res.shape], idx]);
    // }
};
