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
        // console.log('Raw zarr.json metadata:', metadata);
        this.metadata = metadata;
        
        // Debug the actual metadata for init_time and lead_time
        if (metadata.consolidated_metadata?.metadata) {
          // No action needed, metadata is fine
        } else {
          console.warn('No consolidated_metadata.metadata found');
        }
        
        // Check for different metadata structures
        this.arrays = {};
        if (metadata.meta) {
          for (const [name, arrayMeta] of Object.entries(metadata.meta)) {
            this.arrays[name] = this.createArrayAccessor(name, arrayMeta);
          }
        } else if (metadata.consolidated_metadata) {
                  // Check if consolidated_metadata has a nested metadata structure
        if (metadata.consolidated_metadata.metadata) {
          for (const [name, arrayMeta] of Object.entries(metadata.consolidated_metadata.metadata)) {
            // Create a deep copy of the metadata to avoid sharing references
            const metadataCopy = structuredClone ? structuredClone(arrayMeta) : JSON.parse(JSON.stringify(arrayMeta));
            metadataCopy.name = name; // Add the array name to metadata
            
            this.arrays[name] = this.createArrayAccessor(name, metadataCopy);
          }
        } else {
          // Direct consolidated_metadata structure
          for (const [name, arrayMeta] of Object.entries(metadata.consolidated_metadata)) {
            // Create a deep copy of the metadata to avoid sharing references
            const metadataCopy = structuredClone ? structuredClone(arrayMeta) : JSON.parse(JSON.stringify(arrayMeta));
            metadataCopy.name = name; // Add the array name to metadata
            this.arrays[name] = this.createArrayAccessor(name, metadataCopy);
          }
        }
        } else {
          console.error('No recognizable metadata structure found');
          callback(new Error('No recognizable metadata structure found'));
          return;
        }
        
        callback(null, this.arrays);
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
    // Get the array accessor to get the correct metadata
    const arrayAccessor = this.arrays[name];
    if (!arrayAccessor) {
      console.error(`Array ${name} not found in arrays`);
      callback(new Error(`Array ${name} not found`));
      return;
    }
    
    const arrayMeta = arrayAccessor.metadata;
    
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
    // Get the array accessor to get the correct metadata
    const arrayAccessor = this.arrays[name];
    if (!arrayAccessor) {
      console.error(`Array ${name} not found in arrays`);
      callback(new Error(`Array ${name} not found`));
      return;
    }
    
    const arrayMeta = arrayAccessor.metadata;
    
    // Convert slice to chunk coordinates
    const chunkCoords = this.sliceToChunkCoords(slice, arrayMeta);
    
    // Load the chunk and extract the specific value
    this.loadArrayChunk(name, chunkCoords, (error, result) => {
      if (error) {
        callback(error);
        return;
      }
      
      // Extract the specific value from the chunk
      const chunkData = result.data;      
      const chunks = arrayMeta.chunk_grid?.configuration?.chunk_shape || [1];
      
      // Calculate the offset within the chunk
      let offset = 0;
      for (let i = 0; i < slice.length; i++) {
        const coord = slice[i];
        const chunkSize = chunks[i];
        const localCoord = coord % chunkSize;
        
        // Calculate offset for this dimension
        let dimOffset = localCoord;
        for (let j = i + 1; j < slice.length; j++) {
          const nextChunkSize = chunks[j];
          dimOffset *= nextChunkSize;
        }
        offset += dimOffset;
      }
            
      // Get the single value at the calculated offset
      const value = chunkData[offset];
      
      callback(null, { data: [value] });
    });
  };

  ZarrBrowser.prototype.sliceToChunkCoords = function(slice, metadata) {
    // Simple implementation - assume slice maps to chunk (0,0,0,0)
    // In a full implementation, you'd calculate the correct chunk coordinates
    return [0, 0, 0, 0];
  };

  ZarrBrowser.prototype.loadArrayChunk = function(name, chunkCoords, callback) {
    // Get the array accessor to get the correct metadata
    const arrayAccessor = this.arrays[name];
    if (!arrayAccessor) {
      console.error(`Array ${name} not found in arrays`);
      callback(new Error(`Array ${name} not found`));
      return;
    }
    
    const arrayMeta = arrayAccessor.metadata;
    
    // Add chunk coordinates to metadata for test data generation
    arrayMeta.requestedChunkCoords = chunkCoords;
    
    // Build the correct chunk path for Zarr v3
    const chunkPath = this.buildChunkPath(name, chunkCoords, arrayMeta);
    
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
            
            console.log(`Creating empty data for ${name}: shape=${shape}, size=${size}, dtype=${dtype}, arrayMeta keys:`, Object.keys(arrayMeta));
            
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
            
            console.log(`Created empty data for ${name}: length=${emptyData.length}`);
            callback(null, { data: emptyData });
            return;
          }
          throw new Error(`Failed to fetch chunk: ${response.status}`);
        }
        
        // Check if we're getting the expected chunk size
        const chunks = arrayMeta.chunk_grid?.configuration?.chunk_shape || arrayMeta.shape;
        const chunkSize = chunks.reduce((a, b) => a * b, 1) * this.getBytesPerElement(arrayMeta.data_type);
        const actualSize = response.headers.get('content-length');
        
        if (actualSize && parseInt(actualSize) < chunkSize) {
          console.warn(`File size mismatch for ${name}: expected chunk size ${chunkSize}, got ${actualSize} bytes`);
        }
        
        return response.arrayBuffer();
      })
      .then(buffer => {
        if (buffer) { // Only parse if we got a buffer (not 404 case)
          const view = new Uint8Array(buffer);
          
          // Check if the metadata indicates compression
          const codecs = arrayMeta.codecs || [];
          const hasCompression = codecs.some(codec => codec.name !== 'bytes');
          
          // Check if the data looks compressed (starts with compression header patterns)
          if ((view[0] === 0x02 && view[1] === 0x01 && view[2] === 0x91 && view[3] === 0x01) ||
              (view[0] === 0x18 && view[1] === 0xab && view[2] === 0x00 && view[3] === 0x00) ||
              (view[0] === 0xa8 && view[1] === 0x05 && view[2] === 0x00 && view[3] === 0x00)) {
            console.warn(`Data for ${name} appears to be compressed, attempting to skip header`);
            // Skip the first 4 bytes (compression header) and use the rest
            buffer = buffer.slice(4);
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
    // Use chunk shape if available, otherwise fall back to full shape
    const chunks = metadata.chunk_grid?.configuration?.chunk_shape || metadata.shape || [1];
    const size = chunks.reduce((a, b) => a * b, 1);
    const dtype = metadata.data_type || 'int64';
    
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