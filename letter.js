// // import * as tf from '@tensorflow/tfjs';
// class tf_op_layer_strided_slice extends tf.layers.Layer {
//     constructor() {
//         super({});
//     }
//     computeOutputShape() {
//         console.log('strided one');
//         return [1,128];
//     }
//     async call(input_3) {
//         // this.invokeCallHook(input_3)
//         console.log('first strided slice');
//         stride = tf.stridedSlice(input_3,[0,0,0,0],[0,1,0,1],[1,1,1,1],begin_mask=5, end_mask=5,
//                                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=10);
//         return stride;
//     }
//     static get className() {
//         return 'TensorFlowOpLayer';
//     }
// }
// tf.serialization.registerClass(tf_op_layer_strided_slice);

// const model = tf.loadLayersModel('warp_generator/model.json');
