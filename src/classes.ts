export interface MaskDetectionClass {
    name: string;
    id: number;
    displayName: string;
  }
  
  export const CLASSES: {[key: string]: MaskDetectionClass} = {
    1: {
      name: 'person',
      id: 1,
      displayName: 'person',
    },
    2: {
      name: 'mask',
      id: 2,
      displayName: 'person with mask',
    }
  };