slice_width = 0.19634954084936207 // pi/16

velocity_colors = [
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
  request.open("GET", files[index], true);
  request.responseType = "arraybuffer";

  request.onload = function () {
    if (request.response) {
      var byteArray = new Uint8Array(request.response);
      parser(byteArray);
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
      var color = velocity_colors[max_speed];
      var theta = -3.14159 + 0.39269908 * dir;
      var polygon = L.polygon([
          [lat, lon],
          RadialPoint(lat, lon, theta - slice_width, radius),
          RadialPoint(lat, lon, theta, radius),
          RadialPoint(lat, lon, theta + slice_width, radius)
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
  return {'lat': lat, 'lon': lon, 'speeds': speeds, 'directions': directions}
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
  return `data/${lon}_${lat}.bin`;
}

function CountMatrix(bytes) {

    var parsed = DecodePackedSpot(bytes);
    
    var counts = []
        
    for (let bin = 0; bin < velocity_colors.length; bin++) {
      for (let i = 0; i < parsed['hours'].length; i++) {
        var n = parsed['speeds'][i].length;
        var count = 0;
        for (let j = 0; j < n; j++) {        
          if (parsed['speeds'][i][j] == bin) {
            count += 1;
          }
        }
        counts.push({'bin': bin, 'hour': i, 'count': count});
      }
    }
    return counts;
}

function PopulateHeatMap(bytes, div) {
  var data = CountMatrix(bytes)
  
  // set the dimensions and margins of the graph
  var margin = {top: 30, right: 30, bottom: 30, left: 30}
  width = 450 - margin.left - margin.right,
  height = 450 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = div
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

  var bin_min = d3.min(data, function(d) { return d.bin;});
  var bin_max = d3.max(data, function(d) {
      if (d.count > 0) {
        return d.bin;
      } else {
        return 0;
      }});
  
  bins =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  hours = [0, 1, 2, 3, 4, 5];
  
  var hour_min = 0;
  var hour_max = d3.max(data, function(d) { return d.hour;});

  // Build X scales and axis:
  var x = d3.scaleBand()
    .range([ 0, width ])
    .domain(hours)
    .padding(0.01);

  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x))

  // Build X scales and axis:
  var y = d3.scaleBand()
    .range([ height, 0 ])
    .domain(bins)
    .padding(0.01);

  svg.append("g")
    .call(d3.axisLeft(y));

  // Build color scale
  var myColor = d3.scaleLinear()
    .range(["white", "#69b3a2"])
    .domain([1,31])

  svg.selectAll()
      .data(data, function(d) {return d.hour+':'+d.bin;})
      .enter()
      .append("rect")
      .attr("x", function(d) { return x(d.hour) })
      .attr("y", function(d) { return y(d.bin) })
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) {
        return myColor(d.count)
      } )
}


function SpotPlot(x) {

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

  components.push(outline)
  return components;
}

function ParseSlocum(data) {
  var packet_size = DecodeUint32(data.subarray(0, 4));
  data = data.subarray(4);
  console.log("packed size: ", packet_size)
  console.log("num points: ", data.length / packet_size)
  
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
  var radius = grid_size / 5;

  circles = [];
  for (i in data) {
    var pt = data[i];
    var use_it = zoom >= 8 || (pt['lat'] % grid_size == 0 && pt['lon'] % grid_size == 0);
    if (use_it) {
      if (pt['lat'] <= bounds.getNorth() &&
          pt['lat'] >= bounds.getSouth() &&
          pt['lon'] <= bounds.getEast() &&
          pt['lon'] >= bounds.getWest()) {
        next_circle = BuildWindCircle(pt['lat'], pt['lon'], pt['speeds'], pt['directions'], radius)
        circles = circles.concat(next_circle);
      }
    }
  }
  return circles
}
