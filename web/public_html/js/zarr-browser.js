// Browser-compatible wrapper for zarr-js
// This provides a simple interface to load Zarr v3 datasets in the browser

(function(window) {
  'use strict';

  // Simple Zarr v3 loader for browser
  function ZarrBrowser() {
    this.baseUrl = null;
    this.metadata = null;
    this.arrays = {};
  }

  ZarrBrowser.prototype.open = function(baseUrl, callback) {
    this.baseUrl = baseUrl;
    
    // Load zarr.json metadata
    fetch(`${baseUrl}/zarr.json`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch zarr.json: ${response.status}`);
        }
        return response.json();
      })
      .then(metadata => {
        this.metadata = metadata;
        console.log('Loaded zarr v3 metadata:', metadata);
        console.log('Metadata keys:', Object.keys(metadata));
        
        // Check for different metadata structures
        let arrays = {};
        if (metadata.meta) {
          console.log('Found metadata.meta:', metadata.meta);
          for (const [name, arrayMeta] of Object.entries(metadata.meta)) {
            arrays[name] = this.createArrayAccessor(name, arrayMeta);
          }
        } else if (metadata.consolidated_metadata) {
          console.log('Found metadata.consolidated_metadata:', metadata.consolidated_metadata);
          
                  // Check if consolidated_metadata has a nested metadata structure
        if (metadata.consolidated_metadata.metadata) {
          console.log('Found nested metadata.consolidated_metadata.metadata:', metadata.consolidated_metadata.metadata);
          for (const [name, arrayMeta] of Object.entries(metadata.consolidated_metadata.metadata)) {
            arrayMeta.name = name; // Add the array name to metadata
            arrays[name] = this.createArrayAccessor(name, arrayMeta);
          }
        } else {
          // Direct consolidated_metadata structure
          for (const [name, arrayMeta] of Object.entries(metadata.consolidated_metadata)) {
            arrayMeta.name = name; // Add the array name to metadata
            arrays[name] = this.createArrayAccessor(name, arrayMeta);
          }
        }
        } else {
          console.error('No recognizable metadata structure found');
          console.log('Available keys:', Object.keys(metadata));
          callback(new Error('No recognizable metadata structure found'));
          return;
        }
        
        console.log('Created arrays:', Object.keys(arrays));
        callback(null, arrays);
      })
      .catch(error => {
        console.error('Failed to open zarr dataset:', error);
        callback(error);
      });
  };

  ZarrBrowser.prototype.createArrayAccessor = function(name, metadata) {
    const self = this;
    
    return {
      name: name,
      metadata: metadata,
      
      // Get data from the array
      get: function(slice, callback) {
        if (typeof slice === 'function') {
          callback = slice;
          slice = null;
        }
        
        if (slice === null) {
          // Load entire array
          self.loadEntireArray(name, callback);
        } else {
          // Load specific slice
          self.loadArraySlice(name, slice, callback);
        }
      }
    };
  };

  ZarrBrowser.prototype.loadEntireArray = function(name, callback) {
    // Check all possible metadata structures
    let arrayMeta = this.metadata.meta ? this.metadata.meta[name] : null;
    if (!arrayMeta && this.metadata.consolidated_metadata) {
      if (this.metadata.consolidated_metadata.metadata) {
        arrayMeta = this.metadata.consolidated_metadata.metadata[name];
      } else {
        arrayMeta = this.metadata.consolidated_metadata[name];
      }
    }
    
    if (!arrayMeta) {
      console.error(`Array ${name} not found in metadata`);
      console.log('Available in meta:', this.metadata.meta ? Object.keys(this.metadata.meta) : 'none');
      console.log('Available in consolidated_metadata:', this.metadata.consolidated_metadata ? Object.keys(this.metadata.consolidated_metadata) : 'none');
      if (this.metadata.consolidated_metadata && this.metadata.consolidated_metadata.metadata) {
        console.log('Available in consolidated_metadata.metadata:', Object.keys(this.metadata.consolidated_metadata.metadata));
      }
      callback(new Error(`Array ${name} not found`));
      return;
    }
    
    console.log(`Loading entire array: ${name}`, arrayMeta);
    
    // For 1D arrays, load the first chunk
    // For multi-dimensional arrays, we need to load all chunks and merge them
    const shape = arrayMeta.shape || [1];
    if (shape.length === 1) {
      // 1D array - load the first chunk
      this.loadArrayChunk(name, [0], callback);
    } else {
      // Multi-dimensional array - for now, load the first chunk of each dimension
      // This is a simplified approach - in a full implementation you'd load all chunks
      const chunkCoords = new Array(shape.length).fill(0);
      this.loadArrayChunk(name, chunkCoords, callback);
    }
  };

  ZarrBrowser.prototype.loadArraySlice = function(name, slice, callback) {
    // Check all possible metadata structures
    let arrayMeta = this.metadata.meta ? this.metadata.meta[name] : null;
    if (!arrayMeta && this.metadata.consolidated_metadata) {
      if (this.metadata.consolidated_metadata.metadata) {
        arrayMeta = this.metadata.consolidated_metadata.metadata[name];
      } else {
        arrayMeta = this.metadata.consolidated_metadata[name];
      }
    }
    
    if (!arrayMeta) {
      console.error(`Array ${name} not found in metadata`);
      callback(new Error(`Array ${name} not found`));
      return;
    }
    
    console.log(`Loading array slice: ${name} at ${slice}`, arrayMeta);
    
    // Convert slice to chunk coordinates
    const chunkCoords = this.sliceToChunkCoords(slice, arrayMeta);
    this.loadArrayChunk(name, chunkCoords, callback);
  };

  ZarrBrowser.prototype.sliceToChunkCoords = function(slice, metadata) {
    // Simple implementation - assume slice maps to chunk (0,0,0,0)
    // In a full implementation, you'd calculate the correct chunk coordinates
    return [0, 0, 0, 0];
  };

  ZarrBrowser.prototype.loadArrayChunk = function(name, chunkCoords, callback) {
    // Check all possible metadata structures
    let arrayMeta = this.metadata.meta ? this.metadata.meta[name] : null;
    if (!arrayMeta && this.metadata.consolidated_metadata) {
      if (this.metadata.consolidated_metadata.metadata) {
        arrayMeta = this.metadata.consolidated_metadata.metadata[name];
      } else {
        arrayMeta = this.metadata.consolidated_metadata[name];
      }
    }
    
    if (!arrayMeta) {
      callback(new Error(`Array ${name} not found in metadata`));
      return;
    }
    
    // Add chunk coordinates to metadata for test data generation
    arrayMeta.requestedChunkCoords = chunkCoords;
    
    // Build the correct chunk path for Zarr v3
    const chunkPath = this.buildChunkPath(name, chunkCoords, arrayMeta);
    
    console.log(`Loading chunk: ${chunkPath}`);
    console.log(`Array metadata for ${name}:`, arrayMeta);
    
    fetch(chunkPath, {
      headers: {
        'Accept-Encoding': 'identity' // Disable compression
      }
    })
      .then(response => {
        if (!response.ok) {
          if (response.status === 404) {
            // Chunk file doesn't exist - this might be normal for some arrays
            console.warn(`Chunk file not found: ${chunkPath}`);
            // Return empty data based on the array shape
            const shape = arrayMeta.shape || [1];
            const size = shape.reduce((a, b) => a * b, 1);
            const dtype = arrayMeta.data_type || 'int64';
            
            // Create empty data of the right type and size
            let emptyData;
            if (dtype === 'int64' || dtype === '<i8') {
              emptyData = new BigInt64Array(size);
            } else if (dtype === 'int32' || dtype === '<i4') {
              emptyData = new Int32Array(size);
            } else if (dtype === 'uint8' || dtype === '<u1') {
              emptyData = new Uint8Array(size);
            } else if (dtype === 'uint16' || dtype === '<u2') {
              emptyData = new Uint16Array(size);
            } else if (dtype === 'float64' || dtype === '<f8') {
              emptyData = new Float64Array(size);
            } else if (dtype === 'float32' || dtype === '<f4') {
              emptyData = new Float32Array(size);
            } else {
              emptyData = new Uint8Array(size);
            }
            
            console.log(`Created empty data for ${name}: ${emptyData.length} elements`);
            callback(null, { data: emptyData });
            return;
          }
          throw new Error(`Failed to fetch chunk: ${response.status}`);
        }
        
        console.log(`Response headers for ${name}:`, {
          'content-length': response.headers.get('content-length'),
          'content-encoding': response.headers.get('content-encoding'),
          'accept-ranges': response.headers.get('accept-ranges'),
          'content-type': response.headers.get('content-type'),
          'server': response.headers.get('server'),
          'date': response.headers.get('date')
        });
        
        // Check if we're getting the full file
        const expectedSize = arrayMeta.shape.reduce((a, b) => a * b, 1) * this.getBytesPerElement(arrayMeta.data_type);
        const actualSize = response.headers.get('content-length');
        console.log(`Expected size for ${name}: ${expectedSize} bytes, actual: ${actualSize} bytes`);
        
        if (actualSize && parseInt(actualSize) < expectedSize) {
          console.warn(`File size mismatch for ${name}: expected ${expectedSize}, got ${actualSize}`);
        }
        
        return response.arrayBuffer();
      })
      .then(buffer => {
        if (buffer) { // Only parse if we got a buffer (not 404 case)
          console.log(`Received buffer for ${name}: ${buffer.byteLength} bytes`);
          
          // Debug: show first few bytes of the buffer
          const view = new Uint8Array(buffer);
          console.log(`First 32 bytes of ${name}:`, Array.from(view.slice(0, 32)).map(b => b.toString(16).padStart(2, '0')).join(' '));
          
          // Check if the data looks compressed (starts with 02 01 91 01 pattern)
          if (view[0] === 0x02 && view[1] === 0x01 && view[2] === 0x91 && view[3] === 0x01) {
            console.warn(`Data appears to be compressed for ${name}, attempting to skip compression header`);
            // Skip the first 4 bytes (compression header) and use the rest
            buffer = buffer.slice(4);
            console.log(`After skipping header, buffer size: ${buffer.byteLength} bytes`);
          }
          
          // Parse the chunk data based on metadata
          const data = this.parseChunkData(buffer, arrayMeta);
          callback(null, { data: data });
        }
      })
      .catch(error => {
        console.error(`Failed to load chunk for ${name}:`, error);
        callback(error);
      });
  };

  ZarrBrowser.prototype.buildChunkPath = function(name, chunkCoords, arrayMeta) {
    // For Zarr v3, chunks are stored in nested directories under 'c/'
    // For 1D arrays like init_time, the path is: {name}/c/{chunkId}
    // For multi-dimensional arrays, the path is: {name}/c/{dim0}/{dim1}/{dim2}/.../{chunkFile}
    
    const shape = arrayMeta.shape || [1];
    const chunks = arrayMeta.chunk_grid?.configuration?.chunk_shape || shape;
    
    console.log(`Building chunk path for ${name}: shape=${shape}, chunks=${chunks}, coords=${chunkCoords}`);
    
    if (shape.length === 1) {
      // 1D array - simple case
      return `${this.baseUrl}/${name}/c/0`;
    } else {
      // Multi-dimensional array - calculate chunk indices
      const chunkIndices = [];
      for (let i = 0; i < chunkCoords.length; i++) {
        const coord = chunkCoords[i];
        const chunkSize = chunks[i];
        const chunkIndex = Math.floor(coord / chunkSize);
        chunkIndices.push(chunkIndex);
      }
      
      // For 5D arrays like speed/direction, we need to add the chunk file number at the end
      // The chunk file number is typically the last coordinate
      const chunkFile = chunkCoords[chunkCoords.length - 1] % chunks[chunks.length - 1];
      chunkIndices.push(chunkFile);
      
      console.log(`Chunk indices for ${name}: ${chunkIndices}`);
      return `${this.baseUrl}/${name}/c/${chunkIndices.join('/')}`;
    }
  };

  ZarrBrowser.prototype.getBytesPerElement = function(dtype) {
    switch (dtype) {
      case 'int64':
      case '<i8':
      case 'float64':
      case '<f8':
        return 8;
      case 'int32':
      case '<i4':
      case 'float32':
      case '<f4':
        return 4;
      case 'uint16':
      case '<u2':
        return 2;
      case 'uint8':
      case '<u1':
        return 1;
      default:
        console.warn(`Unknown dtype: ${dtype}, assuming 1 byte`);
        return 1;
    }
  };

  ZarrBrowser.prototype.parseChunkData = function(buffer, metadata) {
    // Parse the actual binary chunk data (no compression)
    const shape = metadata.shape || [1];
    const size = shape.reduce((a, b) => a * b, 1);
    const dtype = metadata.data_type || 'int64';
    
    console.log(`Parsing chunk data: shape=${shape}, size=${size}, dtype=${dtype}, buffer size=${buffer.byteLength}`);
    
    // Parse the binary data based on dtype (no decompression needed)
    let data;
    try {
      if (dtype === 'int64' || dtype === '<i8') {
        // Handle partial buffers for int64
        const bytesPerElement = 8;
        const maxElements = Math.floor(buffer.byteLength / bytesPerElement);
        const actualElements = Math.min(size, maxElements);
        
        if (actualElements === 0) {
          console.warn(`Buffer too small for int64 data, creating empty array`);
          data = new BigInt64Array(size);
        } else {
          // Create a view of the buffer with the available data
          const view = new DataView(buffer, 0, actualElements * bytesPerElement);
          data = new BigInt64Array(actualElements);
          for (let i = 0; i < actualElements; i++) {
            data[i] = view.getBigInt64(i * bytesPerElement, true); // true for little-endian
          }
          
          // If we got fewer elements than expected, pad with zeros
          if (actualElements < size) {
            const fullData = new BigInt64Array(size);
            fullData.set(data);
            data = fullData;
          }
        }
      } else if (dtype === 'int32' || dtype === '<i4') {
        const bytesPerElement = 4;
        const maxElements = Math.floor(buffer.byteLength / bytesPerElement);
        const actualElements = Math.min(size, maxElements);
        
        if (actualElements === 0) {
          data = new Int32Array(size);
        } else {
          const view = new DataView(buffer, 0, actualElements * bytesPerElement);
          data = new Int32Array(actualElements);
          for (let i = 0; i < actualElements; i++) {
            data[i] = view.getInt32(i * bytesPerElement, true);
          }
          
          if (actualElements < size) {
            const fullData = new Int32Array(size);
            fullData.set(data);
            data = fullData;
          }
        }
      } else if (dtype === 'uint8' || dtype === '<u1') {
        data = new Uint8Array(buffer);
      } else if (dtype === 'uint16' || dtype === '<u2') {
        const bytesPerElement = 2;
        const maxElements = Math.floor(buffer.byteLength / bytesPerElement);
        const actualElements = Math.min(size, maxElements);
        
        if (actualElements === 0) {
          data = new Uint16Array(size);
        } else {
          const view = new DataView(buffer, 0, actualElements * bytesPerElement);
          data = new Uint16Array(actualElements);
          for (let i = 0; i < actualElements; i++) {
            data[i] = view.getUint16(i * bytesPerElement, true);
          }
          
          if (actualElements < size) {
            const fullData = new Uint16Array(size);
            fullData.set(data);
            data = fullData;
          }
        }
      } else if (dtype === 'float64' || dtype === '<f8') {
        const bytesPerElement = 8;
        const maxElements = Math.floor(buffer.byteLength / bytesPerElement);
        const actualElements = Math.min(size, maxElements);
        
        if (actualElements === 0) {
          data = new Float64Array(size);
        } else {
          const view = new DataView(buffer, 0, actualElements * bytesPerElement);
          data = new Float64Array(actualElements);
          for (let i = 0; i < actualElements; i++) {
            data[i] = view.getFloat64(i * bytesPerElement, true);
          }
          
          if (actualElements < size) {
            const fullData = new Float64Array(size);
            fullData.set(data);
            data = fullData;
          }
        }
      } else if (dtype === 'float32' || dtype === '<f4') {
        const bytesPerElement = 4;
        const maxElements = Math.floor(buffer.byteLength / bytesPerElement);
        const actualElements = Math.min(size, maxElements);
        
        if (actualElements === 0) {
          data = new Float32Array(size);
        } else {
          const view = new DataView(buffer, 0, actualElements * bytesPerElement);
          data = new Float32Array(actualElements);
          for (let i = 0; i < actualElements; i++) {
            data[i] = view.getFloat32(i * bytesPerElement, true);
          }
          
          if (actualElements < size) {
            const fullData = new Float32Array(size);
            fullData.set(data);
            data = fullData;
          }
        }
      } else {
        console.warn(`Unknown dtype: ${dtype}, using Uint8Array`);
        data = new Uint8Array(buffer);
      }
      
      console.log(`Successfully parsed chunk data: ${data.length} elements`);
      return data;
    } catch (error) {
      console.error(`Failed to parse chunk data:`, error);
      throw new Error(`Failed to parse chunk data: ${error.message}`);
    }
  };



ZarrBrowser.prototype.generateTestData = function(shape, dtype, metadata) {
  // This method is kept for compatibility but should not be used for real data loading
  throw new Error('Test data generation disabled - use real data loading');
};

  // Create the main zarr function
  function createZarr(request, version, config) {
    if (version === 'v3') {
      return new ZarrBrowser();
    } else {
      throw new Error(`Version ${version} not supported in browser wrapper`);
    }
  }

  // Expose to global scope
  window.zarr = createZarr;

})(window); 