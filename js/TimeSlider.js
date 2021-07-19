Date.prototype.addHours = function(h) {
  this.setTime(this.getTime() + (h*60*60*1000));
  return this;
}

L.Control.TimeSlider = L.Control.extend({
    options: {
        position: 'topright',
        layers: null,
        timeAttribute: 'time',
        isEpoch: false,     // whether the time attribute is seconds elapsed from epoch
        startTimeIdx: 0,    // where to start looking for a timestring
        timeStrLength: 19,  // the size of  yyyy-mm-dd hh:mm:ss - if millis are present this will be larger
        maxValue: null,
        minValue: 0,
        showAllOnStart: false,
        markers: [],
        range: false,
        follow: false,
        sameDate: false,
        alwaysShowDate : true,
        rezoom: null,
        refTime: null,
        hours: null
    },

    initialize: function (options) {
        L.Util.setOptions(this, options);
    },

    extractTimestamp: function(i, options) {
        var time = new Date(options.refTime)
        time.addHours(options.hours[i])
        return time.toString();
    },

    setPosition: function (position) {
        var map = this._map;

        if (map) {
            map.removeControl(this);
        }

        this.options.position = position;

        if (map) {
            map.addControl(this);
        }
        this.startSlider();
        return this;
    },

    onAdd: function (map) {
        this.options.map = map;

        var options = this.options;

        // Create a control sliderContainer with a jquery ui slider
        var sliderContainer = L.DomUtil.create('div', 'slider', this._container);
        $(sliderContainer).append('<div id="leaflet-slider" style="width:600px"><div class="ui-slider-handle"></div><div id="slider-timestamp" style="width:600px; margin-top:13px; background-color:#FFFFFF; text-align:center; border-radius:5px;"></div></div>');
        //Prevent map panning/zooming while using the slider
        $(sliderContainer).mousedown(function () {
            map.dragging.disable();
        });
        $(document).mouseup(function () {
            map.dragging.enable();
            //Hide the slider timestamp if not range and option alwaysShowDate is set on false
            if (options.range || !options.alwaysShowDate) {
                $('#slider-timestamp').html('');
            }
            updateValidTime();
        });

        //If a layer has been provided: calculate the min and max values for the slider
        options.maxValue = options.hours.length - 1;

        this.options = options;
        return sliderContainer;
    },

    onRemove: function (map) {
        //Delete all markers which where added via the slider and remove the slider div
        for (i = this.options.minValue; i <= this.options.maxValue; i++) {
            //map.removeLayer(this.options.markers[i]);
        }
        $('#leaflet-slider').remove();

        // unbind listeners to prevent memory leaks
        $(document).off("mouseup");
        $(".slider").off("mousedown");
    },

    startSlider: function () {
        _options = this.options;
        _extractTimestamp = this.extractTimestamp
        var index_start = _options.minValue;
        if(_options.showAllOnStart){
            index_start = _options.maxValue;
            if(_options.range) _options.values = [_options.minValue,_options.maxValue];
            else _options.value = _options.maxValue;
        }
        $("#leaflet-slider").slider({
            range: _options.range,
            value: _options.value,
            values: _options.values,
            min: _options.minValue,
            max: _options.maxValue,
            sameDate: _options.sameDate,
            step: 1,
            slide: function (e, ui) {
                var map = _options.map;
                var fg = L.featureGroup();
                
                $('#slider-timestamp').html(
                                _extractTimestamp(ui.value, _options));
                HOUR = _options.hours[ui.value];
                if(_options.rezoom) {
                    map.fitBounds(fg.getBounds(), {
                        maxZoom: _options.rezoom
                    });
                }
            }
        });
        //if (!_options.range && _options.alwaysShowDate) {
        //    $('#slider-timestamp').html(_extractTimeStamp(ui.value, _options));
        //}
    }
});

L.control.timeSlider = function (options) {
    return new L.Control.TimeSlider(options);
};
