/* Polyfill for maplibre-gl class field transformation (Vite/esbuild __publicField) */
/* Use var to create a global variable so bundled code can reference __publicField */
var __publicField = function (obj, key, value) {
  Object.defineProperty(obj, key, { enumerable: true, configurable: true, writable: true, value });
};
