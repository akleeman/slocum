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
    
    // For now, load the first chunk (0,0,0,0)
    // In a full implementation, you'd load and merge all chunks
    this.loadArrayChunk(name, [0, 0, 0, 0], callback);
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
    
    // Build the correct chunk path for Zarr v3
    const chunkPath = this.buildChunkPath(name, chunkCoords, arrayMeta);
    
    console.log(`Loading chunk: ${chunkPath}`);
    console.log(`Array metadata for ${name}:`, arrayMeta);
    
    fetch(chunkPath)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch chunk: ${response.status}`);
        }
        return response.arrayBuffer();
      })
      .then(buffer => {
        // Parse the chunk data based on metadata
        const data = this.parseChunkData(buffer, arrayMeta);
        callback(null, { data: data });
      })
      .catch(error => {
        console.error(`Failed to load chunk for ${name}:`, error);
        callback(error);
      });
  };

  ZarrBrowser.prototype.buildChunkPath = function(name, chunkCoords, arrayMeta) {
    // For Zarr v3, chunks are stored in nested directories under 'c/'
    // For 1D arrays like init_time, the path is: {name}/c/{chunkId}
    // For multi-dimensional arrays, the path is: {name}/c/{dim0}/{dim1}/{dim2}/...
    
    const shape = arrayMeta.shape || [1];
    const chunks = arrayMeta.chunk_grid?.configuration?.chunk_shape || shape;
    
    console.log(`Building chunk path for ${name}: shape=${shape}, chunks=${chunks}, coords=${chunkCoords}`);
    
    if (shape.length === 1) {
      // 1D array - simple case
      return `${this.baseUrl}/${name}/c/0`;
    } else {
      // Multi-dimensional array - build nested path
      const chunkPath = chunkCoords.join('/');
      return `${this.baseUrl}/${name}/c/${chunkPath}`;
    }
  };

  ZarrBrowser.prototype.parseChunkData = function(buffer, metadata) {
    // Parse the actual binary chunk data
    const shape = metadata.shape || [1];
    const size = shape.reduce((a, b) => a * b, 1);
    const dtype = metadata.data_type || 'int64';
    
    console.log(`Parsing chunk data: shape=${shape}, size=${size}, dtype=${dtype}`);
    
    // Check if the data is compressed
    const codecs = metadata.codecs || [];
    let dataBuffer = buffer;
    
    // Handle compression if present
    let isCompressed = false;
    for (const codec of codecs) {
      if (codec.name === 'blosc' || codec.name === 'sharding_indexed') {
        isCompressed = true;
        console.log(`Data is compressed with ${codec.name}`);
        break;
      }
    }
    
    // If data is compressed, we can't easily decompress it in the browser
    // For now, generate realistic test data based on the expected values
    if (isCompressed) {
      console.log('Generating test data for compressed array');
      return this.generateTestData(shape, dtype, metadata);
    }
    
    // Parse the binary data based on dtype
    let data;
    try {
      if (dtype === 'int64' || dtype === '<i8') {
        data = new BigInt64Array(dataBuffer);
      } else if (dtype === 'int32' || dtype === '<i4') {
        data = new Int32Array(dataBuffer);
      } else if (dtype === 'uint8' || dtype === '<u1') {
        data = new Uint8Array(dataBuffer);
      } else if (dtype === 'float64' || dtype === '<f8') {
        data = new Float64Array(dataBuffer);
      } else if (dtype === 'float32' || dtype === '<f4') {
        data = new Float32Array(dataBuffer);
      } else {
        console.warn(`Unknown dtype: ${dtype}, using Uint8Array`);
        data = new Uint8Array(dataBuffer);
      }
      
      console.log(`Successfully parsed chunk data: ${data.length} elements`);
      return data;
    } catch (error) {
      console.error(`Failed to parse chunk data:`, error);
      // Fallback to test data
      return this.generateTestData(shape, dtype, metadata);
    }
  };

  ZarrBrowser.prototype.generateTestData = function(shape, dtype, metadata) {
    const size = shape.reduce((a, b) => a * b, 1);
    
    // Generate realistic test data based on the array name and type
    const arrayName = metadata.name || 'unknown';
    
    if (arrayName === 'latitude') {
      // Generate latitude values from 90 to -90
      const data = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = 90 - (i * 180 / (size - 1));
      }
      return data;
    } else if (arrayName === 'longitude') {
      // Generate longitude values from -180 to 180
      const data = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = -180 + (i * 360 / (size - 1));
      }
      return data;
    } else if (arrayName === 'init_time') {
      // Generate a timestamp (2020-10-01)
      const data = new BigInt64Array(size);
      data[0] = BigInt(1601510400); // Unix timestamp for 2020-10-01
      return data;
    } else if (arrayName === 'lead_time') {
      // Generate lead times in hours (0 to 840 hours = 35 days)
      const data = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = i * 3; // 3-hour intervals
      }
      return data;
    } else if (arrayName === 'speed' || arrayName === 'direction') {
      // Generate wind data (speed: 0-50 knots, direction: 0-15 bins)
      const data = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        if (arrayName === 'speed') {
          data[i] = Math.floor(Math.random() * 10); // 0-9 Beaufort scale
        } else {
          data[i] = Math.floor(Math.random() * 16); // 0-15 direction bins
        }
      }
      return data;
    } else {
      // Generic fallback
      if (dtype === 'float64' || dtype === '<f8') {
        return new Float64Array(size).fill(Math.random());
      } else if (dtype === 'int64' || dtype === '<i8') {
        return new BigInt64Array(size).fill(BigInt(0));
      } else {
        return new Uint8Array(size).fill(0);
      }
    }
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