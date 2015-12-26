Introduction
===========================

Sending a forecast query.
~~~~~~~~~~~~~~~~~~

Slocum uses a query syntax that is similar to saildocs.  If you're familiar with these, using slocum will be easy.

Here are some examples:

``send GFS:35N,45N,120W,130W|1,1|0,6..144|WIND``  will send a single gridded forecast from the GFS model which
covers a ten by ten degree area around San Francisco.  The forecast will be 1 degree resolution and contain
a wind forecast every six hours for the next six days.

``send spot:32N,-117E|8,6|wind`` will send a `spot forecast <spot.html>`_ just off the coast of San Diego that
has a wind forecast for every six hours for the next eight days.


What inspired slocum?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
While crossing the Pacific in our boat `Saltbreaker <http://www.saltbreaker.com>`_ we would use our single
side band radio and a modem to download weather forecasts using `saildocs <http://www.saildocs.com/>`_.
As you can imagine the connection was *extremely* slow (4,000 bits/minute on a good day) leading to
downloads that could take 15-30 minutes and draining precious battery power.

At some point I looked at what exactly was being sent in a grib file and realized that

::
    from slocum.query import saildocs
    print saildocs._send_usage


Why the name slocum?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
| `Joshua Slocum <http://en.wikipedia.org/wiki/Joshua_Slocum>`_ (February 20, 1844 -on or shortly after November 14, 1909) was a Canadian-American seaman and adventurer, a noted writer, and the first man to sail single-handedly around the world. In 1900 he told the story of this in Sailing Alone Around the World. He disappeared in November 1909 while aboard his boat, the Spray. [1]_
|

In `Sailing Alone Around the World <http://en.wikipedia.org/wiki/Sailing_Alone_Around_the_World>`_ he
talks about his tin clock,

| My tin clock and only timepiece had by this time lost its minute-hand, but after I boiled her she told the hours, and that was near enough on a long stretch.
|

If Joshua Slocum was able to successfully circumnavigate via celestial navigation using a clock with no minute hand, then modern day sailors should be able to get by using weather forecasts without floating point precision.


Getting Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slocum is (primarily) intended to be used as an e-mail based forecast provider and supports most of the same
`query syntax <http://www.saildocs.com/gribinfo>`_ as saildocs and zygrib, if you are familiar with either of
those simply redirect one of your requsts to query@ensembleweather.com.

To obtain a forecast you send an e-mail to query@ensembleweather.com with a forecast request in the body, then wait a few minutes for an e-mail response.  The respone will contain an attachment which holds an ultra-compressed forecast.  You then run the slocum program on your computer to decompress and plot the forecast.
For details see the page on <a href="email.html">e-mail based forecasts</a>, roughly the process goes as follows

* Create an e-mail with an e-mail request in the body.
* Send the e-mail to query@ensembleweather.com.
* `Grab a cup of coffee <http://media.giphy.com/media/AOjF59lD6eOPe/giphy.gif>`_ and wait for a response.
* The response e-mail contains a forecast file attached, save it to your computer.
* Plot the forecast using slocum by opening a `terminal <http://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_ and running ``slocum ./path_to_forecast.fcst``.


You'll notice that this is identical to how sailmail or zygrib work with one exception,
because slocum uses a unique compression algorithm you'll need slocum installed on your computer
in order to open the forecasts.  In otherwords, you will not be able to use a grib viewer
to view the forecast so be sure you have slocum install before cutting your dock lines.


.. [1] `wikipedia <http://en.wikipedia.org/wiki/Joshua_Slocum>`_