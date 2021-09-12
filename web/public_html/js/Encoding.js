var SLICE_WIDTH = 0.19634954084936207 // pi/16

var WIND_BINS = [0., 1., 3., 6., 10., 16., 21., 27.,
             33., 40., 47., 55., 63., 75.]

var VELOCITY_COLORS = [
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
  request.open("GET", path, true);
  request.responseType = "arraybuffer";
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

function fitToBounds(data, bounds) {
  var mid = 0.5 * (bounds.getWest() + bounds.getEast())
  for (i in data) {
    var offset = Math.round((data[i]['lon'] - mid) / 360.)
    data[i]['lon'] -= offset * 360;
  }
  return data;
}

