var HOUR = 0;

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

function DecodeFloat32(x) {
  var buf = new ArrayBuffer(4);
  var view = new DataView(buf);

  for (let i = 0; i < 4; i++) {
    view.setUint8(i, x[i]);
  };
  return view.getFloat32(0, true);
}

function DecodeUint32(x) {
  var buf = new ArrayBuffer(4);
  var view = new DataView(buf);

  for (let i = 0; i < 4; i++) {
    view.setUint8(i, x[i]);
  };
  return view.getUint32(0, true);
}

function LoadFile(path, parser) {
  var result = null;
  var request = new XMLHttpRequest();
  request.open("GET", path, true);
  request.responseType = "arraybuffer";
  console.log(path)
  request.onload = function () {
    if (request.readyState==4 &&
        request.status==200 &&
        request.response) {
      var byteArray = new Uint8Array(request.response);
      parser(byteArray);
    } else {
      console.log("failure loading:", path)
      console.log(request)
    }
  };

  request.send();
}

function LoadSpot(path, parser) {
  var result = null;
  var request = new XMLHttpRequest();
  request.open("GET", path, true);
  request.responseType = "arraybuffer";
  request.onload = function () {
    if (request.response) {
      var byteArray = new Uint8Array(request.response);
      parser(byteArray);
    }    
  };

  request.send();
}


function RadialPoint(x, y, angle, radius) {
  return [x + Math.cos(3.1415 * x / 180) * radius * Math.cos(angle),
          y + radius * Math.sin(angle)]
}


function DrawSlices(lat, lon, speeds, directions, radius) {
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
          RadialPoint(lat, lon, theta - SLICE_WIDTH, radius),
          RadialPoint(lat, lon, theta, radius),
          RadialPoint(lat, lon, theta + SLICE_WIDTH, radius)
      ], {color: color, fillOpacity:1., stroke:false});
      slices.push(polygon);
    }
  }
  return slices;
}


function DecodePackedPoint(x) {
  lon = DecodeFloat32(x.subarray(0, 4))
  lat = DecodeFloat32(x.subarray(4, 8))
  var n_members = (x.length - 8) / 2
  speeds = x.subarray(8, 8 + n_members);
  directions = x.subarray(8 + n_members, 8 + 2 * n_members);
  return {'lat': lat, 'lon': lon - 360, 'speeds': speeds, 'directions': directions}
}


function DecodePackedSpot(x) {
  n_times = x[0]
  diffs = x.subarray(1, n_times);
  hours = [0];
  for (let i = 0; i < n_times - 1; i++) {
    hours.push(hours[i] + diffs[i])
  };

  x = x.subarray(n_times);
  var n_members = (x.length / (2 * n_times))

  speeds = []; 
  directions = [];
  for (let i = 0; i < n_times; i++) {
      speeds.push(x.subarray(n_members * i, n_members * (i + 1)));
      directions.push(x.subarray(n_members * (i + n_times), n_members * (i + 1 + n_times)));
  }
  return {'hours': hours, 'speeds': speeds, 'directions': directions}
}

function SpotPath(lat, lon) {
  lat = lat.toFixed(3);
  lon = lon.toFixed(3);
  console.log(`${lat}_${lon}.bin`)
  return `data/spot/${lon}_${lat}.bin`;
}

function binnedData(bytes) {

    var parsed = DecodePackedSpot(bytes);
    var data = []
    for (let bin = 0; bin < WIND_BINS.length; bin++) {
      for (let i = 0; i < parsed['hours'].length; i++) {
        var n = parsed['speeds'][i].length;
        var directions = []
        for (let j = 0; j < n; j++) {        
          if (parsed['speeds'][i][j] == bin) {
            directions.push(parsed['directions'][i][j])
          }
        }
        data.push({'speed': WIND_BINS[bin], 'hour': parsed['hours'][i], 'directions': directions});
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

function PopulateHeatMap(bytes, div) {
  var data = binnedData(bytes)

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
  var days = new Set()
  var tickValues = new Set()
  for (i in data) {
    speeds.add(data[i].speed)
    days.add(data[i].hour)
    tickValues.add(data[i].hour - data[i].hour % 24.)
  }
  
  speeds = Array.from(speeds);
  days = Array.from(days);
  tickValues = Array.from(tickValues);

  tickFormat = function(x) {
    return x / 24.;
  };

  // Build X scales and axis:
  var x = d3.scaleBand()
    .range([ 0, width ])
    .domain(days)
    .padding(0.01);

  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x)
          .tickValues(tickValues)
          .tickFormat(tickFormat))

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
        return x(d.hour) + 0.5 * x.bandwidth()
      })
      .attr("y", function(d) {
        return y(d.speed) - 0.5 * y.bandwidth()
      })
      .attr("width", x.bandwidth() )
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
  
  x.options.color = "#CC5500"

  var div = d3.create("div");

  LoadSpot(SpotPath(x._latlng.lat, x._latlng.lng), function (x) {
    PopulateHeatMap(x, div);
  })

  return div.node();
}

function BuildWindCircle(lat, lon, speeds, directions, radius) {
  var radiusMeters = L.latLng(lat, lon).distanceTo(L.latLng(lat, lon + radius))

  // White Background Circle
  var circle = L.circle([lat, lon], radiusMeters, {
        stroke: false,
        fillColor: '#FFFFFF',
        fillOpacity: 0.9
    })

  var components = [circle];

  var slices = DrawSlices(lat, lon, speeds, directions, radius)
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

  outline.on("click", function (x) {
    console.log(x)
    
    var div = d3.select("#id")
    LoadSpot(SpotPath(x.latlng.lat, x.latlng.lng), function (x) {
    PopulateHeatMap(x, div);
  })
    
  });

  components.push(outline)
  return components;
}

function ParseSlocum(data) {
  var packet_size = DecodeUint32(data.subarray(0, 4));
  data = data.subarray(4);  
  decoded_data = []
  
  var count = 0;
  while (data.length > 0) {
    count += 1;
    one_point = data.subarray(0, packet_size);
    var output = DecodePackedPoint(one_point)
    decoded_data.push(output);
    data = data.subarray(packet_size);
  }
  return decoded_data;
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
  return circles
}



function GetRadius(zoom) {
  if (zoom <= 3) {
    radius = 2.56;
  } else if (zoom <= 4) {
    radius = 1.28;
  } else if (zoom <= 5) {
    radius = 0.64;
  } else if (zoom <= 6) {
    radius = 0.32;
  } else if (zoom <= 7) {
    radius = 0.16;
  } else {
    radius = 0.08
  }
  return radius
}


function Draw(m, data) {
  var bounds = m.getBounds();
  var zoom = m.getZoom();

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
  
  var radius = GetRadius(zoom);
  var radius = grid_size / 4;

  var only_visible = [];
  for (i in data) {
    var pt = data[i];
    var on_grid = (pt['lat'] % grid_size == 0 && pt['lon'] % grid_size == 0);
    if (on_grid && InView(bounds, pt)) {
      only_visible.append(pt);
    }
  }
  return DrawAll(only_visible, radius);
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
        
        const layer = L.layerGroup().addTo(map);
        this.m_tileLayers[id] = layer;
    
        path = `./data/${coords.z}/${HOUR}/${coords.x}_${coords.y}.bin`;
 
        LoadFile(path,  function (bytes) {
          var data = ParseSlocum(bytes);
          circles = DrawAll(ParseSlocum(bytes), GetRadius(coords.z));
          L.layerGroup(circles).addTo(layer);
          done(error, tile);
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


function parseForecastTimes(bytes) {
  var timestamp = DecodeUint32(bytes.subarray(0, 4))
  bytes = bytes.subarray(4);
  // Create a new JavaScript Date object based on the timestamp
  // multiplied by 1000 so that the argument is in milliseconds, not seconds.
  REF_TIME = new Date(timestamp * 1000);
  ms_per_minute = 60*1000
  offset = REF_TIME.getTimezoneOffset() * ms_per_minute;
  REF_TIME = new Date(REF_TIME - offset);
  
  var n_times = DecodeUint32(bytes.subarray(0, 4));
  var bytes = bytes.subarray(4);
  var diffs = bytes.subarray(0, n_times);
  FCST_HOURS = [0];
  for (let i = 0; i < n_times - 1; i++) {
    FCST_HOURS.push(FCST_HOURS[i] + diffs[i])
  };
}

function LoadForecastTimes(map) {
    path = `./data/time.bin`;
    
    parseAndLoad = function(bytes) {
      parseForecastTimes(bytes);
      
      sliderControl = L.control.timeSlider({
        refTime: REF_TIME,
        hours: FCST_HOURS
      });

      //Make sure to add the slider to the map ;-)
      map.addControl(sliderControl);
      //And initialize the slider
      sliderControl.startSlider();
    }
    
    LoadFile(path, parseAndLoad)
}


function initializeSlocum(map) {

  LoadForecastTimes(map)

  var layer = windCircleLayer();
  map.addLayer( layer );
  
}

function updateValidTime() {
  map.removeLayer(L.GridLayer.WindCircles);
  map.addLayer(windCircleLayer());
}

function incrementValidTime() {
  HOUR += 6;
  updateValidTime()
}
