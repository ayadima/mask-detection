import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import * as native from '@tensorflow/tfjs-react-native'

export {version} from './version';
import {CLASSES} from './classes';


export interface DetectedMask {
  bbox: [number, number, number, number];  // [x, y, width, height]
  class: string;
  score: number;
}

export async function load(path : string, modelJson? : tf.io.ModelJSON, modelWeights? : number) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  const maskDetection = new MaskDetection(path);
  if(modelJson && modelWeights){
    await maskDetection.load(modelJson,modelWeights)
  } else {
    await maskDetection.load();
  }
  return maskDetection;
}

export async function loadNative(path : string) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  const maskDetection = new MaskDetection(path);
  await maskDetection.loadNativeModel();
  return maskDetection;
}

export class MaskDetection {
  private modelPath: string;
  private model: tfconv.GraphModel;

  constructor(path : string) {
    this.modelPath = path;
  }

  async load(modelJson? : tf.io.ModelJSON, modelWeights? : number) {
    if(modelJson && modelWeights){
      this.model = await tfconv.loadGraphModel(native.bundleResourceIO(modelJson, modelWeights))
    } else {
      tf.enableProdMode()
      tf.setBackend('webgl')
      tf.webgl.forceHalfFloat()
      this.model = await tfconv.loadGraphModel(this.modelPath);
    }
        // Warmup the model.
        const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as
        tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
  }

  async loadNativeModel() {
    this.model = await tfconv.loadGraphModel(native.asyncStorageIO(this.modelPath));
    // Warmup the model.
    const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as
        tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
  }


  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   */
  private async infer(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement): Promise<DetectedMask[]> {
        
    tf.enableProdMode()
    tf.setBackend('webgl')
    tf.webgl.forceHalfFloat()
    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0);
    });

    const height = batched.shape[1];
    const width = batched.shape[2];

    let boxes_index;
    let scores_index;
    let classes_index;


    for (var i=0; i< this.model.outputs.length; i++){
      if(this.model.outputs[i].name === 'detection_boxes'){
        boxes_index = i
      } else if (this.model.outputs[i].name === 'detection_scores'){
        scores_index = i
      } else if (this.model.outputs[i].name === 'detection_classes'){
        classes_index = i
      }
    }

    // model returns three main tensors:
    // 1. box classification score 
    // 2. box location 
    // 3. box detected classes
    const result = await this.model.executeAsync(batched) as tf.Tensor[];


    const boxes = result[boxes_index].dataSync() as Float32Array;
    const scores = result[scores_index].dataSync() as Float32Array;
    const detection_classes = result[classes_index].dataSync() as Float32Array;

    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);


    const prevBackend = tf.getBackend();
    // run post process in cpu
    tf.setBackend('cpu');
    const indexTensor = tf.tidy(() => {
      const boxes2 =
          tf.tensor2d(boxes, [result[boxes_index].shape[1], result[boxes_index].shape[2]]);
      return tf.image.nonMaxSuppression(
          boxes2, scores,100, 0.6, 0.6);
    });

      const indexes = indexTensor.dataSync() as Float32Array;
      indexTensor.dispose();

      // restore previous backend
      tf.setBackend(prevBackend);
      return this.buildDetectedObjects(
          width, height, boxes, scores, indexes, detection_classes);
      }

private buildDetectedObjects(
  width: number, height: number, boxes: Float32Array, scores: Float32Array,
  indexes: Float32Array, classes: Float32Array): DetectedMask[] {
    const count = indexes.length;
    const objects: DetectedMask[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox as [number, number, number, number],
        class: CLASSES[classes[indexes[i]]].displayName,
        score: scores[indexes[i]]
      });
    }
    return objects;
}


  /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   *
   */
  async detect(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement): Promise<DetectedMask[]> {
    return this.infer(img);
  }

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
