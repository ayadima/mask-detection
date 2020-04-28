export interface MaskDetectionClass {
    name: string;
    id: number;
    displayName: string;
  }
  
  export const CLASSES: {[key: string]: MaskDetectionClass} = {
    2: {
      name: 'person',
      id: 1,
      displayName: 'person',
    },
    1: {
      name: 'mask',
      id: 2,
      displayName: 'person with mask',
    }
  };