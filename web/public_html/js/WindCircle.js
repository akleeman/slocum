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
				    // empty image
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
        
        var bounds = this._map.getBounds();
        LoadFile(path,  function (bytes) {
          var data = ParseSlocum(bytes);
          data = fitToBounds(data, bounds);
          circles = DrawAll(data, GetRadius(coords.z));
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

