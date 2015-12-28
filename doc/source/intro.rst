Introduction
===========================

Getting Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slocum is (primarily) intended to be used as an e-mail based forecast provider and supports most of the same
query syntax as `saildocs <http://www.saildocs.com/gribinfo>`_ and `zygrib <http://www.zygrib.org/?page=gribauto>`_,
if you are familiar with either of those simply redirect one of your requsts to query@ensembleweather.com.
For more details see the page on :doc:`query syntax <queries>`.

Here is a basic example:

* Create an e-mail with a query request (such as ``send GEFS:35N,45N,120W,130W|native|0,6..144|WIND``) in the body.
* Send the e-mail to query@ensembleweather.com.
* `Grab a cup of coffee <http://media.giphy.com/media/AOjF59lD6eOPe/giphy.gif>`_ and wait for a response.
* The response e-mail will have a forecast file attached, download it.
* :ref:`Run slocum <run-slocum>` using the forecast file to plot it.

You'll notice that this is similar to how sailmail or zygrib work with one exception,
because slocum uses a unique compression algorithm (instead of less efficient grib files)
you'll need slocum installed on your computer in order to open the forecasts.  In otherwords,
you will not be able to use a grib viewer to view the forecast so be sure you have slocum
install before cutting your dock lines.

What inspired slocum?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
While crossing the Pacific in our boat `Saltbreaker <http://www.saltbreaker.com>`_ we would use our single
side band radio and a modem to download weather forecasts through `saildocs <http://www.saildocs.com/>`_.
The connection was *extremely* slow (4,000 bits/minute on a good day) leading to
downloads that could take 15-30 minutes and draining precious battery power.

At some point I looked at what exactly was being sent in a grib file and noticed the files held
far more information than sailors need to know.  For example, we don't need to know if the wind
speed will be 17.54367 knots, knowing it will be between 15 and 20 is perfectly sufficient.  This
inspired the compression algorithm used by slocum which sends the `Beaufort Force <https://en.wikipedia.org/wiki/Beaufort_scale>`_
number (instead of floating point decimal values) resulting in forecast files that are an
order of magnitude smaller than grib files.  A similar scheme is used to send other weather
variables important to sailors.


Why the name slocum?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
| `Joshua Slocum <http://en.wikipedia.org/wiki/Joshua_Slocum>`_ (February 20, 1844 -on or shortly after November 14, 1909) was a Canadian-American seaman and adventurer, a noted writer, and the first man to sail single-handedly around the world. In 1900 he told the story of this in Sailing Alone Around the World. He disappeared in November 1909 while aboard his boat, the Spray. [1]_
|

In `Sailing Alone Around the World <http://en.wikipedia.org/wiki/Sailing_Alone_Around_the_World>`_ he
talks about his tin clock,

| My tin clock and only timepiece had by this time lost its minute-hand, but after I boiled her she told the hours, and that was near enough on a long stretch.
|

If Joshua Slocum was able to successfully circumnavigate via celestial navigation using a clock with no minute hand, then modern day sailors should be able to get by using weather forecasts without floating point precision.


.. [1] `wikipedia <http://en.wikipedia.org/wiki/Joshua_Slocum>`_
