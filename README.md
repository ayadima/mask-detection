# Mask Detection

Mask detection model that aims to localize, identify and distinguish workers wearing security masks from those not wearing security masks in a single image.


This TensorFlow.js model does not require you to know about machine learning.
It can take input as any browser-based image elements (`<img>`, `<video>`, `<canvas>`
elements, for example) and returns an array of bounding boxes with class name and confidence level.

## Usage

There are one main way to get this model in your JavaScript project : by installing it from NPM and using a build tool like Parcel, WebPack, or Rollup.

### via NPM

```js
// Note: you do not need to import @tensorflow/tfjs here.

import * as mask from 'maskdetection';

const img = document.getElementById('img');

// Load the model.
const model = await mask.load(PATH_TO_JSON_MODEL);

// Classify the image.
const predictions = await model.detect(img);

console.log('Predictions: ');
console.log(predictions);
```

## API

#### Loading the model
`maskdetection` is the module name. When using ES6 imports, `mask` is the module.

```ts

mask.load(PATH_TO_JSON_MODEL);
```

Args:
**PATH_TO_JSON_MODEL** string that specifies json file as input of the model. This file can be an url or a locally stored file.

Returns a `model` object.

#### Detecting workers 

You can detect workers wearing masks and those who are not with the model without needing to create a Tensor.
`model.detect` takes an input image element and returns an array of bounding boxes with class name and confidence level.

This method exists on the model that is loaded from `mask.load`.

```ts
model.detect(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement
)
```

Args:

**img:** A Tensor or an image element to make a detection on.

Returns an array of classes and probabilities that looks like:

```js
[{
  bbox: [x, y, width, height],
  class: "person",
  score: 0.8380282521247864
}, {
  bbox: [x, y, width, height],
  class: "person with mask",
  score: 0.74644153267145157
}]
```
