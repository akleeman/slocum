var HOUR = 0;
console.log('Initial HOUR value:', HOUR);

SLICE_WIDTH = 0.19634954084936207 // pi/16

WIND_BINS = [0., 1., 3., 6., 10., 16., 21., 27.,
             33., 40., 47., 55., 63., 75.]

VELOCITY_COLORS = [
    '#d7d7d7',  // light grey
    '#a1eeff',  // lightest blue
    '#42b1e5',  // light blue
    '#4277e5',  // pastel blue
    '#60fd4b',  // green
    '#1cea00',  // yellow-green
    '#fbef36',  // yellow
    '#fbc136',  // orange
    '#ff4f02',  // red
    '#d50c02',  // darker-red
    '#ff00c0',  // red-purple
    '#b30d8a',  // dark purple
    '#000000',  // black
]

var REF_TIME = undefined;
var FCST_HOURS = undefined;

// Zarr configuration
var ZARR_BASE_URL = 'http://example.com/data.zarr';  // This will be configurable
var zarrStore = null;
var zarrDataset = null;

Date.prototype.addHours = function(h) {
  this.setTime(this.getTime() + (h*60*60*1000));
  return this;
}

// Load zarr library with Zarr v3 support
function loadZarrLibrary(callback) {
  if (typeof window.zarr === 'undefined') {
    throw new Error('zarr-js library not loaded. Check console for import errors.');
  }
  
  // Initialize zarr-js with fetch and v3 support
  window.zarrInstance = window.zarr(window.fetch, 'v3');
  console.log('zarr-js v3 library ready');
  callback();
}

// Initialize zarr dataset using zarr-js v3 API
async function initializeZarr(baseUrl) {
  try {
    if (!window.zarrInstance) {
      throw new Error('zarr-js v3 library not loaded');
    }
    
    // Convert relative URL to absolute URL
    let absoluteUrl = baseUrl || ZARR_BASE_URL;
    if (absoluteUrl.startsWith('./') || absoluteUrl.startsWith('../')) {
      // Convert relative path to absolute URL
      const baseUrl = window.location.origin + window.location.pathname.replace(/\/[^\/]*$/, '/');
      absoluteUrl = baseUrl + absoluteUrl.replace(/^\.\//, '');
    }
    
    ZARR_BASE_URL = absoluteUrl;
    console.log('Original URL:', baseUrl || ZARR_BASE_URL);
    console.log('Absolute URL:', ZARR_BASE_URL);
    console.log('Initializing zarr v3 dataset from:', ZARR_BASE_URL);
    
    // Use zarr-js v3 open method to get array accessors
    return new Promise((resolve, reject) => {
      window.zarrInstance.open(ZARR_BASE_URL, (err, arrays) => {
        if (err) {
          console.error('Failed to open zarr v3 dataset:', err);
          reject(new Error(`Failed to open zarr dataset at ${ZARR_BASE_URL}: ${err.message}`));
          return;
        }
        
        console.log('Successfully opened zarr v3 dataset');
        console.log('Available arrays:', Object.keys(arrays));
        
        
        // Store the arrays for later use
        zarrDataset = arrays;
        
        // Log detailed information about the dataset
        console.log('=== ZARR DATASET DETAILS ===');
        for (const [key, array] of Object.entries(arrays)) {
          console.log(`Array: ${key}`);
          console.log(`  Type: Array accessor`);
        }
        console.log('=== END DATASET DETAILS ===');
        
        // Extract forecast times from zarr metadata
        initializeForecastTimesFromZarr()
          .then(() => {
            console.log('Zarr dataset initialized successfully');
            resolve(true);
          })
          .catch(reject);
      });
    });
    
  } catch (error) {
    console.error('Failed to initialize zarr dataset:', error);
    console.error('URL attempted:', ZARR_BASE_URL);
    console.error('Full error details:', error);
    throw new Error(`Failed to open zarr dataset at ${ZARR_BASE_URL}: ${error.message}`);
  }
}

// Extract forecast times from zarr v3 using zarr-js API
async function initializeForecastTimesFromZarr() {
  try {
    console.log('Extracting forecast times from zarr v3...');
    console.log('Available arrays:', Object.keys(zarrDataset));
    
    // Check if init_time and lead_time exist
    if (!zarrDataset.init_time) {
      console.warn('init_time not found in zarr dataset, using current time');
      REF_TIME = new Date();
    } else {
      console.log('Found init_time array');
      
      try {
        // Load init_time data using zarr-js v3 API
        const initTimeArray = await loadZarrArrayV3('init_time');
        console.log('Raw init_time data:', initTimeArray);
        
        // Handle different data formats
        let refTime;
        if (Array.isArray(initTimeArray)) {
          if (initTimeArray.length > 0 && initTimeArray[0] !== 0) {
            refTime = new Date(initTimeArray[0] * 1000); // Convert seconds to milliseconds
          } else {
            console.warn('init_time array is empty or contains zeros, using current time');
            refTime = new Date();
          }
        } else {
          if (initTimeArray !== 0) {
            refTime = new Date(initTimeArray * 1000);
          } else {
            console.warn('init_time value is zero, using current time');
            refTime = new Date();
          }
        }
        REF_TIME = refTime;
        console.log('Parsed REF_TIME:', REF_TIME);
      } catch (error) {
        console.warn('Failed to load init_time data, using current time:', error);
        REF_TIME = new Date();
      }
    }
    
    if (!zarrDataset.lead_time) {
      throw new Error('lead_time not found in zarr dataset');
    }
    
    console.log('Found lead_time array');
    const leadTimeArray = await loadZarrArrayV3('lead_time');
    console.log('Raw lead_time data:', leadTimeArray);
    
    // Handle different data formats
    let fcstHours;
    if (Array.isArray(leadTimeArray)) {
      fcstHours = Array.from(leadTimeArray);
    } else {
      fcstHours = [leadTimeArray];
    }
    
    FCST_HOURS = fcstHours;
    console.log('Parsed FCST_HOURS:', FCST_HOURS);
    console.log('Successfully set REF_TIME to:', REF_TIME);
    console.log('Successfully set FCST_HOURS to:', FCST_HOURS);
    
  } catch (error) {
    console.error('Failed to extract forecast times:', error);
    throw new Error(`Failed to extract forecast times: ${error.message}`);
  }
}

// Convert Beaufort scale back to wind speeds
function fromBeaufort(beaufortValue) {
  const divs = WIND_BINS;
  const ndivs = divs.length;
  const bins = beaufortValue;
  const upperBins = Math.min(bins, ndivs - 1);
  const lowerBins = Math.max(upperBins - 1, 0);
  const average = 0.5 * (divs[lowerBins] + divs[upperBins]);
  
  // Handle edge cases
  const epsilon = Math.max(0.5 * Math.min(...divs.slice(1).map((v, i) => v - divs[i])), 1e-6);
  if (bins === 0) return Math.min(...divs) - epsilon;
  if (bins === ndivs) return Math.max(...divs) + epsilon;
  
  return average;
}

// Helper function to load Zarr v3 arrays using zarr-js v3 API
async function loadZarrArrayV3(arrayName) {
  try {
    console.log(`Loading zarr v3 array: ${arrayName}`);
    
    if (!zarrDataset[arrayName]) {

      throw new Error(`Array ${arrayName} not found in dataset`);
    }
    
    // Use zarr-js v3 API to get all data from the array
    return new Promise((resolve, reject) => {
      zarrDataset[arrayName].get(null, (err, data) => {
        if (err) {
          console.error(`Failed to get data from array ${arrayName}:`, err);
          reject(err);
          return;
        }
        
        resolve(data.data);
      });
    });
    
  } catch (error) {
    console.error(`Failed to load zarr array ${arrayName}:`, error);
    throw error;
  }
}

// Helper function to load a slice of a Zarr v3 array using zarr-js v3 API
async function loadZarrArraySliceV3(arrayName, slice) {
  try {
    console.log(`Loading zarr v3 array slice: ${arrayName} at ${slice}`);
    
    if (!zarrDataset[arrayName]) {
      throw new Error(`Array ${arrayName} not found in dataset`);
    }
    
    // Use zarr-js v3 API to get the specific slice
    return new Promise((resolve, reject) => {
      zarrDataset[arrayName].get(slice, (err, data) => {
        if (err) {
          console.error(`Failed to get slice from array ${arrayName}:`, err);
          reject(err);
          return;
        }
        
        console.log(`Successfully loaded array slice ${arrayName}:`, data);
        resolve(data.data);
      });
    });
    
  } catch (error) {
    console.error(`Failed to load zarr array slice ${arrayName}:`, error);
    throw error;
  }
}

// Convert integer direction bins back to radians
function fromIntegerDirection(directionBin) {
  const dirBins = Array.from({length: 16}, (_, i) => -15 * Math.PI/16 + i * Math.PI/8);
  const bins = directionBin;
  const upper = bins % dirBins.length;
  const wrapVal = -Math.PI;
  const lower = (bins - 1) % dirBins.length;
  
  if (bins > 0) {
    return 0.5 * (dirBins[lower] + dirBins[upper]);
  } else {
    return wrapVal;
  }
}

// Load wind data from zarr v3 for a specific tile and time
async function loadZarrWindData(tileCoords, leadTimeIndex) {
  try {
    if (!zarrDataset) {
      throw new Error('Zarr dataset not initialized');
    }
    
    console.log('=== WIND DATA LOADING DEBUG ===');
    console.log('Loading zarr wind data for tile:', tileCoords, 'time index:', leadTimeIndex);
    
    // Calculate tile bounds using Leaflet's map.unproject, ensuring we match
    // the grid that Leaflet actually displays for this tile.
    const TILE_PX = 512; // Must match windCircleLayer's tileSize
    const nw = map.unproject([tileCoords.x * TILE_PX, tileCoords.y * TILE_PX], tileCoords.z);
    const se = map.unproject([(tileCoords.x + 1) * TILE_PX, (tileCoords.y + 1) * TILE_PX], tileCoords.z);

    const lonMin = nw.lng;
    const lonMax = se.lng;
    const latMax = nw.lat;
    const latMin = se.lat;
    
    console.log(`Tile bounds: lat [${latMin}, ${latMax}], lon [${lonMin}, ${lonMax}]`);
    
    // Load coordinate arrays
    let latArray, lonArray;
    try {
      [latArray, lonArray] = await Promise.all([
        loadZarrArrayV3('latitude'),
        loadZarrArrayV3('longitude')
      ]);
      
      console.log(`Loaded coordinates: ${latArray.length} lats, ${lonArray.length} lons`);
      console.log('Latitude range:', latArray[0], 'to', latArray[latArray.length-1]);
      console.log('Longitude range:', lonArray[0], 'to', lonArray[lonArray.length-1]);
      
      // Check if the data is corrupted (tiny numbers indicate corrupted data)
      const latMin = Math.abs(latArray[0]);
      const latMax = Math.abs(latArray[latArray.length-1]);
      const lonMin = Math.abs(lonArray[0]);
      const lonMax = Math.abs(lonArray[lonArray.length-1]);
      
      if (latMin < 1e-100 || latMax < 1e-100 || lonMin < 1e-100 || lonMax < 1e-100) {
        throw new Error('Coordinate data appears to be corrupted (tiny numbers detected)');
      }
      
    } catch (error) {
      console.warn('Failed to load coordinates from zarr, using hardcoded values:', error);
      
      // Use hardcoded coordinate arrays for testing
      latArray = [];
      for (let i = 0; i < 721; i++) {
        latArray.push(90 - i * 0.25); // 90 to -90 in 0.25 degree steps
      }
      
      lonArray = [];
      for (let i = 0; i < 1440; i++) {
        lonArray.push(-180 + i * 0.25); // -180 to 179.75 in 0.25 degree steps
      }
      
      console.log(`Using hardcoded coordinates: ${latArray.length} lats, ${lonArray.length} lons`);
      console.log('Latitude range:', latArray[0], 'to', latArray[latArray.length-1]);
      console.log('Longitude range:', lonArray[0], 'to', lonArray[lonArray.length-1]);
    }
    
    // Find indices within tile bounds
    const validPoints = [];
    for (let latIdx = 0; latIdx < latArray.length; latIdx++) {
      const lat = latArray[latIdx];
      if (lat >= latMin && lat <= latMax) {
        for (let lonIdx = 0; lonIdx < lonArray.length; lonIdx++) {
          const lon = lonArray[lonIdx];
          const normalizedLon = lon > 180 ? lon - 360 : lon;
          if (normalizedLon >= lonMin && normalizedLon <= lonMax) {
            validPoints.push({ lat, lon, latIdx, lonIdx });
          }
        }
      }
    }
    
    console.log(`Found ${validPoints.length} points in tile bounds`);
    if (validPoints.length > 0) {
      console.log('Sample valid points:', validPoints.slice(0, 3));
    }
    
    if (validPoints.length === 0) {
      console.log('No valid points found in tile bounds');
      return [];
    }
    
    // Determine desired spatial thinning based on zoom-level grid spacing so
    // we keep a regular pattern (handled later in Draw).  Here we keep *all*
    // grid-aligned points to ensure a proper lattice, even if that is more
    // than 20.
    const datasetSpacing = 0.1; // degrees between original grid points
    const zoomGridSize = GetGridSize(tileCoords.z);
    const stride = Math.max(1, Math.round(zoomGridSize / datasetSpacing));

    const limitedPoints = validPoints.filter(pt => (pt.latIdx % stride === 0) && (pt.lonIdx % stride === 0));
    console.log(`Filtered ${validPoints.length} valid points down to ${limitedPoints.length} using stride ${stride}`);
    
    // Load wind data for each valid point using zarr-js v3
    const windDataPromises = limitedPoints.map(async (point) => {
      try {
        console.log(`Loading data for point ${point.lat},${point.lon} (indices: ${point.latIdx},${point.lonIdx})`);
        
        // Load speed and direction data for this point at the given time
        const [speedData, directionData] = await Promise.all([
          loadZarrArraySliceV3('speed', [0, leadTimeIndex, point.latIdx, point.lonIdx]),
          loadZarrArraySliceV3('direction', [0, leadTimeIndex, point.latIdx, point.lonIdx])
        ]);
        
        console.log(`Point ${point.lat},${point.lon}: speed=${speedData}, direction=${directionData}`);
        
        return {
          lat: point.lat,
          lon: point.lon,
          speeds: Array.isArray(speedData) ? speedData : [speedData],
          directions: Array.isArray(directionData) ? directionData : [directionData]
        };
      } catch (error) {
        console.warn(`Failed to load wind data for point ${point.lat},${point.lon}:`, error);
        return null;
      }
    });
    
    const results = await Promise.all(windDataPromises);
    const windData = results.filter(r => r !== null);
    console.log(`Successfully loaded wind data for ${windData.length} points`);
    
    if (windData.length > 0) {
      console.log('Sample wind data:', windData.slice(0, 2));
    }
    
    console.log('=== END WIND DATA LOADING DEBUG ===');
    return windData;
    
  } catch (error) {
    console.error('Failed to load zarr wind data:', error);
    throw error;
  }
}

// Load spot forecast data from zarr v3
async function loadZarrSpotData(lat, lon) {
  try {
    if (!zarrDataset) {
      throw new Error('Zarr dataset not initialized');
    }
    
    console.log('Loading zarr spot forecast data for:', lat, lon);
    
    // Load coordinate arrays to find nearest grid point
    const [latArray, lonArray] = await Promise.all([
      loadZarrArrayV3('latitude'),
      loadZarrArrayV3('longitude')
    ]);
    
    // Find nearest grid point
    let nearestLatIdx = 0;
    let nearestLonIdx = 0;
    let minLatDist = Infinity;
    let minLonDist = Infinity;
    
    for (let i = 0; i < latArray.length; i++) {
      const dist = Math.abs(latArray[i] - lat);
      if (dist < minLatDist) {
        minLatDist = dist;
        nearestLatIdx = i;
      }
    }
    
    for (let i = 0; i < lonArray.length; i++) {
      const dist = Math.abs(lonArray[i] - lon);
      if (dist < minLonDist) {
        minLonDist = dist;
        nearestLonIdx = i;
      }
    }
    
    console.log(`Found nearest grid point: lat=${latArray[nearestLatIdx]}, lon=${lonArray[nearestLonIdx]}`);
    
    const spotData = {
      hours: [...FCST_HOURS],
      speeds: [],
      directions: []
    };
    
    // Load data for all forecast times
    const timePromises = FCST_HOURS.map(async (hour, timeIdx) => {
      try {
        // Load speed and direction data for this time
        const [speedData, directionData] = await Promise.all([
          loadZarrArraySliceV3('speed', [0, timeIdx, nearestLatIdx, nearestLonIdx]),
          loadZarrArraySliceV3('direction', [0, timeIdx, nearestLatIdx, nearestLonIdx])
        ]);
        
        return {
          speeds: Array.isArray(speedData) ? speedData : [speedData],
          directions: Array.isArray(directionData) ? directionData : [directionData]
        };
      } catch (error) {
        console.warn(`Failed to load data for time ${hour}:`, error);
        return { speeds: [], directions: [] };
      }
    });
    
    const timeResults = await Promise.all(timePromises);
    timeResults.forEach(result => {
      spotData.speeds.push(result.speeds);
      spotData.directions.push(result.directions);
    });
    
    console.log(`Successfully loaded spot forecast data for ${FCST_HOURS.length} time steps`);
    return spotData;
    
  } catch (error) {
    console.error('Failed to load zarr spot data:', error);
    throw error;
  }
}


function RadialPoint(lat, lon, angle, radiusKm) {
  // Convert a polar offset (radiusKm, angle) into a new [lat, lon] in degrees.
  // We use a simple equirectangular approximation which is sufficient for small
  // radii: 1 degree latitude ≈ 111 km and longitude scaling with cos(latitude).
  const DEG_PER_KM = 1.0 / 111.0;
  const dLat = radiusKm * DEG_PER_KM * Math.sin(angle);
  const dLon = radiusKm * DEG_PER_KM * Math.cos(angle) / Math.cos(lat * Math.PI / 180.0);
  return [lat + dLat, lon + dLon];
}


function DrawSlices(lat, lon, speeds, directions, radiusKm) {
  slices = [];
  for (let dir = 0; dir < 16; dir++) {
    max_speed = null;
    for (let i = 0; i < speeds.length; i++) {  
      if (directions[i] == dir) {
        max_speed = speeds[i];
      }
    }
    if (max_speed != null) {  
      var color = VELOCITY_COLORS[max_speed];
      var theta = -3.14159 + 0.39269908 * dir;
      var polygon = L.polygon([
          [lat, lon],
          RadialPoint(lat, lon, theta - SLICE_WIDTH, radiusKm),
          RadialPoint(lat, lon, theta, radiusKm),
          RadialPoint(lat, lon, theta + SLICE_WIDTH, radiusKm)
      ], {color: color, fillOpacity:1., stroke:false});
      slices.push(polygon);
    }
  }
  return slices;
}


function validLongitude(lon) {
  lon = lon % 360.
  if (lon < 0) {
    lon += 360.
  }
  return lon
}

function SpotPath(lat, lon) {
  lat = lat.toFixed(3);
  lon = validLongitude(lon % 360.).toFixed(3);
  path = `data/spot/${lon}_${lat}.bin`;
  console.log(path)
  return path
}

function binnedData(spotData) {

    var data = []
    for (let bin = 0; bin < WIND_BINS.length; bin++) {
      for (let i = 0; i < spotData.hours.length; i++) {
        var n = spotData.speeds[i].length;
        var directions = []
        for (let j = 0; j < n; j++) {        
          if (spotData.speeds[i][j] == bin) {
            directions.push(spotData.directions[i][j])
          }
        }
        data.push({'speed': WIND_BINS[bin], 'hour': spotData.hours[i], 'directions': directions});
      }
    }
    return data;
}

function polygonSlice(x, y, angle, radius) {

  point = function(angle) {
    return [x + radius * Math.sin(angle),
            y - radius * Math.cos(angle)];
  }

  var points = [ [x, y],
          point(angle - SLICE_WIDTH),
          point(angle),
          point(angle + SLICE_WIDTH),
      ];

  point_strings = []
  for (value of points) {
    point_strings.push(value.join(','))
  }
  return point_strings.join(' ');
}

function windCircleData(x, y, directions, radius) {

  slices = [];
  var n = directions.length;
  for (let dir = 0; dir < 16; dir++) {
    var count = 0;
    for (let i = 0; i < directions.length; i++) {  
      if (directions[i] == dir) {
        count += 1;
      }
    }
    if (count > 0) {
      var color = '#4277e5';
      var theta = -3.14159 + 0.39269908 * dir;

      var slice = {
        'x': x,
        'y': y,
        'theta': theta,
        'points': polygonSlice(x, y, theta, radius),
        'opacity': count / n
      };
        
      slices.push(slice);
    }
  }
  return slices;
}

function PopulateHeatMap(spotData, div) {
  var data = binnedData(spotData)

  // set the dimensions and margins of the graph
  var margin = {top: 30, right: 30, bottom: 30, left: 30}
  width = 600 - margin.left - margin.right,
  height = 250 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = div
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

  var speed_min = d3.min(data, function(d) { return d.speed;});
  var speed_max = d3.max(data, function(d) {
      if (d.directions.length > 0) {
        return d.speed;
      } else {
        return 0;
      }});
  
  var speeds = new Set()  
  var hours = new Set()
  var tickValues = new Set()
  for (i in data) {
    speeds.add(data[i].speed)
    hours.add(data[i].hour)
    tickValues.add(data[i].hour - data[i].hour % 24.)
  }
  
  speeds = Array.from(speeds);
  hours = Array.from(hours);
  tickValues = Array.from(tickValues);

  tickFormat = function(x) {
    return x / 24.;
  };

  var hour_max = d3.max(data, function(d) {
        return d.hour;
      });
  var date_max = new Date(REF_TIME);
  date_max.addHours(hour_max)
      
  var x = d3.scaleTime()
    .domain([REF_TIME, date_max])
    .range([0, width])
    
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x)
          .ticks(d3.timeDay, 1)
          .tickFormat(d3.timeFormat("%a %d")))

  var y = d3.scaleBand()
    .range([ height, 0 ])
    .domain(speeds)
    .padding(0.01);

  svg.append("g")
    .call(d3.axisLeft(y));
    
  // Build color scale
  var colors = d3.scaleLinear()
    .range(["white", "#4277e5"])
    .domain([1,31])

  rectangles = svg.selectAll('rect')
      .data(data, function(d) {return d.hour+':'+d.speed;})
      .enter()
      .append("rect")
      .attr("x", function(d) {
        var date = new Date(REF_TIME);
        date.addHours(d.hour);
        var val = x(date);
        return val;
      })
      .attr("y", function(d) {
        return y(d.speed) - 0.5 * y.bandwidth()
      })
      .attr("width", width / hours.length)
      .attr("height", y.bandwidth() )
      .style("fill", function(d) {
        return colors(d.directions.length)
      } )
      
  var hover = svg.append("g")
                 .attr("id", "hover");

  var mousemove = function(d) {

    if (d.directions.length > 0) {
    
      radius = 30
      width = 2.5 * radius
      height = width
      x = d3.mouse(this)[0] + 2 * radius
      y = d3.mouse(this)[1] - 2 * radius
    
      var dir_data = windCircleData(x, y, d.directions, radius);

      d3.select('#hover').selectAll('circle').remove()
      d3.select('#hover').append('circle')
        .attr("cx", x)
        .attr("cy", y)
        .attr("r", radius)
        .attr("stroke", "black")
        .attr('fill', 'white')

      d3.select('#hover').selectAll('polygon').remove()
      d3.select('#hover').selectAll('polygon')
        .data(dir_data)
        .enter()
        .append('polygon')
        .attr("points", function(d) {
          return d.points;
        })
        .attr("stroke", null)
        .style("fill", 'steelblue')
        .style("fill-opacity", function (d) {
          return d.opacity;
        })
    }
  }
  var mouseleave = function(d) {
    d3.select('#hover').selectAll('circle').remove()
    d3.select('#hover').selectAll('polygon').remove()
  }

  var click = function(d) {
    HOUR = d.hour;
    updateValidTime()
  }

  rectangles
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave)
    .on("click", click)

  var now = new Date();
  var x_now = x(now);

  svg.append("line")
    .style("stroke", "red")
    .attr('x1', x_now)
    .attr('x2', x_now)
    .attr('y1', 0)
    .attr('y2', height)
    .style('stroke-width', 2)

  svg.append("text")
      .attr("class", "y label")
      .attr("text-anchor", "end")
      .attr("y", 6)
      .attr("dy", ".75em")
      .attr("transform", "rotate(-90)")
      .text("wind speed (knots)");

  svg.append("text")
      .attr("class", "x label")
      .attr("text-anchor", "end")
      .attr("x", width)
      .attr("y", height - 6)
      .text("Forecast Time (days)");
}


function SpotPlot(x) {
  
  var half_edge = GetGridSize(this._map.getZoom()) / 2.;
  
  var square = L.polygon([
          [x._latlng.lat - half_edge, x._latlng.lng - half_edge],
          [x._latlng.lat + half_edge, x._latlng.lng - half_edge],
          [x._latlng.lat + half_edge, x._latlng.lng + half_edge],
          [x._latlng.lat - half_edge, x._latlng.lng + half_edge]
      ], {color: "#cc2900", fillOpacity:0.2, stroke:true});

  square.layerID = "highlight"
  
  L.layerGroup([square]).addTo(map);

  var div = d3.create("div");

  loadZarrSpotData(x._latlng.lat, x._latlng.lng).then(function(spotData) {
    PopulateHeatMap(spotData, div);
  }).catch(function(error) {
    console.error('Failed to load spot data:', error);
    div.append("p").text("Failed to load forecast data");
  });

  return div.node();
}

function BuildWindCircle(lat, lon, speeds, directions, radiusKm) {
  const radiusMeters = radiusKm * 1000; // Leaflet circles expect metres
  
  console.log(`BuildWindCircle: lat=${lat}, lon=${lon}, speeds=${speeds}, directions=${directions}, radiusKm=${radiusKm}, radiusMeters=${radiusMeters}`);

  // White Background Circle
  var circle = L.circle([lat, lon], radiusMeters, {
        stroke: false,
        fillColor: '#FFFFFF',
        fillOpacity: 0.9
    })

  var components = [circle];

  // Draw directional slices using radius in km (geographic degrees handled inside)
  var slices = DrawSlices(lat, lon, speeds, directions, radiusKm)
  components = components.concat(slices)

  // Black Outline
  var outline = L.circle([lat, lon], radiusMeters, {
        stroke: true,
        fill: true,
        fillOpacity: 0.0,
        weight: 2,
        color: "#000000"
    })

  outline.bindPopup(SpotPlot);    

  var remove_highlight = function(e) {
    map.eachLayer(function(l) {
      if (l.layerID == "highlight") {
        map.removeLayer(l)
      }
    });
  }
  
  outline.on("popupclose", remove_highlight)

  components.push(outline)
  return components;
}

function InView(bounds, pt) {
  return (pt['lat'] <= bounds.getNorth() &&
          pt['lat'] >= bounds.getSouth() &&
          pt['lon'] <= bounds.getEast() &&
          pt['lon'] >= bounds.getWest())
}


function DrawAll(data, radius) {
  circles = [];
  for (i in data) {
    var pt = data[i];
    next_circle = BuildWindCircle(pt['lat'], pt['lon'], pt['speeds'], pt['directions'], radius)
    circles = circles.concat(next_circle);
  }
  
  console.log(`Total circles created: ${circles.length}`);
  console.log('=== END DRAWALL DEBUG ===');
  return circles
}



function GetGridSize(zoom) {
  var grid_size = 0.25;
  if (zoom <= 3) {
    grid_size = 8;
  } else if (zoom <= 4) {
    grid_size = 4;
  } else if (zoom <= 5) {
    grid_size = 2;
  } else if (zoom <= 6) {
    grid_size = 1.;
  } else if (zoom < 8) {
    grid_size = 0.5;
  }
  return grid_size
}

function GetRadius(zoom) {
  // Make circles much larger for visibility
  return GetGridSize(zoom) * 50; // 50km radius instead of tiny circles
}


function Draw(m, data) {
  var bounds = m.getBounds();
  var zoom = m.getZoom();

  var grid_size = GetGridSize(zoom);
  
  var radius = GetRadius(zoom);

  console.log('=== DRAW DEBUG ===');
  console.log('Input data length:', data.length);
  console.log('Grid size:', grid_size);
  console.log('Map bounds:', bounds);
  console.log('Sample input data:', data.slice(0, 2));

  var only_visible = [];
  for (i in data) {
    var pt = data[i];
    // Previously we required the point to fall exactly on the grid, which
    // eliminated most (or all) locations due to floating-point rounding.
    // That condition has been removed so any point inside the current map
    // bounds will be rendered.
    if (InView(bounds, pt)) {
      only_visible.push(pt);
    }
  }
  
  console.log('Visible points:', only_visible.length);
  if (only_visible.length > 0) {
    console.log('Sample visible points:', only_visible.slice(0, 2));
  }
  console.log('=== END DRAW DEBUG ===');
  
  return DrawAll(only_visible, radius);
}


function fitToBounds(data, bounds) {
  var mid = 0.5 * (bounds.getWest() + bounds.getEast())
  for (i in data) {
    var offset = Math.round((data[i]['lon'] - mid) / 360.)
    data[i]['lon'] -= offset * 360;
  }
  return data;
}


L.GridLayer.WindCircles = L.GridLayer.extend({

   	initialize: function (options) {
   	  L.GridLayer.prototype.initialize.call(this, options);
  		this.m_tileLayers = {};
  	},

	  _abortLoading: function () {
		  var i, tile;
		  for (i in this._tiles) {
			  if (this._tiles[i].coords.z !== this._tileZoom) {
				  tile = this._tiles[i].el;

				  tile.onload = function() { return false; };
				  tile.onerror = function() { return false; };

          if (tile.id in this.m_tileLayers) {
            map.removeLayer(this.m_tileLayers[tile.id]);
          }

				  if (!tile.complete) {
					  tile.src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';

	          var parent = tile.parentNode;
	          if (parent) {
		          parent.removeChild(tile);
	          }

					  delete this._tiles[i];
				  }
			  }
		  }
	  },

    tileId : function(coords) {
      return `${coords.x}_${coords.y}_${coords.z}`
    },

    cleanup : function(zoom) {
      for (var key in this._tiles) {
        var tile = this._tiles[key]        
        if (tile.coords.z != zoom && tile.el.id in this.m_tileLayers) {
          map.removeLayer(this.m_tileLayers[tile.el.id]);
          delete this.m_tileLayers[tile.el.id];
        }
      }
    },

    createTile: function (coords, done) {
        var error;
        var tile = document.createElement('div');

        const id = this.tileId(coords);
        tile.id = id
        
        console.log('=== TILE CREATION DEBUG ===');
        console.log('Creating tile for coords:', coords);
        console.log('Current HOUR:', HOUR);
        
        const layer = L.layerGroup().addTo(map);
        this.m_tileLayers[id] = layer;
    
        // Use zarr data instead of binary files
        const leadTimeIndex = Math.floor(HOUR / 6); // Convert hour to lead time index
        console.log('Calculated leadTimeIndex:', leadTimeIndex);

        loadZarrWindData(coords, leadTimeIndex).then(function(data) {
          console.log('Tile data loaded, length:', data.length);
          var bounds = map.getBounds();
          data = fitToBounds(data, bounds);
          console.log('After fitToBounds, data length:', data.length);
          circles = Draw(map, data);
          console.log('Created circles:', circles.length);
          if (circles.length > 0) {
            console.log('Sample circle:', circles[0]);
            console.log('Circle properties:', {
              latlng: circles[0].getLatLng(),
              radius: circles[0].getRadius(),
              options: circles[0].options
            });
          }
          L.layerGroup(circles).addTo(layer);
      
          console.log('=== END TILE CREATION DEBUG ===');
          done(error, tile);
        }).catch(function(err) {
          console.error('Failed to load tile data:', err);
          console.log('=== END TILE CREATION DEBUG (ERROR) ===');
          done(err, tile);
        });
                        
        return tile;
    }
    
});

windCircleLayer = function(opts) {
    var output = new L.GridLayer.WindCircles({tileSize: 512,
                                              maxNativeZoom: 8,
                                              minNativeZoom: 4});
    
    return output;
};


function initializeSlocum(map, zarrUrl) {
  loadZarrLibrary(function() {
    initializeZarr(zarrUrl).then(function(success) {
      if (success) {
        // Initialize time slider with zarr-based forecast times
      sliderControl = L.control.timeSlider({
        position: "bottomleft",
        refTime: REF_TIME,
        hours: FCST_HOURS
      });

      map.addControl(sliderControl);
      sliderControl.startSlider();
        
        // Add wind layer
  var layer = windCircleLayer();
        map.addLayer(layer);
        
        console.log('Slocum initialized with zarr data successfully');
      } else {
        console.error('Failed to initialize zarr dataset');
      }
    }).catch(function(error) {
      console.error('Failed to initialize Slocum:', error);
    });
  });
}

function updateValidTime() {
  // Remove existing wind layer
  map.eachLayer(function(layer) {
    if (layer instanceof L.GridLayer.WindCircles) {
      map.removeLayer(layer);
    }
  });
  
  // Add new wind layer with updated time
  var layer = windCircleLayer();
  map.addLayer(layer);
}

function incrementValidTime() {
  const currentIndex = FCST_HOURS.indexOf(HOUR);
  if (currentIndex < FCST_HOURS.length - 1) {
    HOUR = FCST_HOURS[currentIndex + 1];
  } else {
    HOUR = FCST_HOURS[0]; // Wrap around
  }
  updateValidTime();
}
