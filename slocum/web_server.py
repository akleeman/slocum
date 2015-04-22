import cherrypy
import datetime

from io import BytesIO

import poseidon
import windbreaker
from slocum.lib import saildocs


class SlocumQuery(object):
    @cherrypy.expose
    def spot(self):
        return """<html>
          <head></head>
          <body>
            <form method="get" action="spot_request">
              <input type="text" value="32N" name="latitude" />
              <input type="text" value="122W" name="longitude" />
              <input type="text" value="4,3" name="hours" />
              <input type="text" value="wind,prmsl" name="vars" />
              <button type="submit">Request Forecast</button>
            </form>
          </body>
        </html>"""

    @cherrypy.expose
    def spot_request(self,
             latitude, longitude,
             hours='4,3',
             variables=None,
             send_image='false',
             model='gefs'):

        warnings = []
        location = {'latitude': float(latitude),
                    'longitude': float(longitude)}
        assert model in poseidon._models.keys()
        hours, time_warnings = saildocs.parse_times(hours)
        warnings.extend(time_warnings)

        if variables is None:
            variables = []
        else:
            variables = variables.split(',')
        variables, var_warnings = saildocs.validate_variables(variables)
        warnings.extend(var_warnings)

        send_image = send_image.lower() == 'true'

        query = {
            'type': 'spot',
            'model': model,
            'location': location,
            'hours': hours,
            'vars': variables,
            'warnings': warnings,
            'send-image': send_image}
        fcst = windbreaker.query_to_beaufort(query)
        file_fmt = '%Y-%m-%d_%H%m'
        filename = datetime.datetime.today().strftime(file_fmt)
        filename = '_'.join([query['type'], filename])
        return cherrypy.lib.static.serve_fileobj(fileobj=BytesIO(fcst),
                                                 content_type="application/x-download",
                                                 disposition="attachment",
                                                 name=filename)


if __name__ == '__main__':
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.quickstart(SlocumQuery())
