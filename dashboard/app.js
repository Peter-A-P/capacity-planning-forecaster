/* The dashboard, drawn from two data files and nothing else.
 *
 * No framework and no charting library: the page draws five panels, and a library big enough
 * to draw them is bigger than the measurements it would draw. Everything here is SVG built
 * with the DOM API, which also means the content security policy can forbid inline scripts
 * and styles outright rather than carrying an exception for a bundle.
 *
 * Colours live in style.css, never here, so that one palette follows the page into dark mode.
 * Lines take a class (m0, m1, m2) rather than a stroke.
 *
 * The one rule worth knowing before changing anything: this file computes nothing that the
 * backtest could have computed. The rolling mean is the exception, and it mirrors
 * headroom.report.charts.rolling_mean exactly, so the chart in the README and the chart on
 * this page agree at the default window.
 */
"use strict";

(function () {
  var NS = "http://www.w3.org/2000/svg";
  var DAY = 86400000;
  var WIDTH = 880;
  var MARGIN = { top: 16, right: 18, bottom: 28, left: 56 };

  var state = {
    dashboard: null,
    forecast: null,
    node: 0,
    range: "shift",
    nominal: null,
    window: 13,
    service: 0
  };

  /* Small DOM helpers --------------------------------------------------------- */

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) { for (var key in attrs) { node.setAttribute(key, attrs[key]); } }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function svg(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    if (attrs) { for (var key in attrs) { node.setAttribute(key, String(attrs[key])); } }
    return node;
  }

  function byId(id) { return document.getElementById(id); }

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  function day(iso) { return Date.parse(iso + "T00:00:00Z") / DAY; }

  function isNumber(value) { return typeof value === "number" && isFinite(value); }

  /* Formatting ---------------------------------------------------------------- */

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function decimals(value, places) {
    return isNumber(value) ? value.toFixed(places) : "n/a";
  }

  function signed(value, places) {
    if (!isNumber(value)) { return "n/a"; }
    return (value > 0 ? "+" : "−") + Math.abs(value).toFixed(places);
  }

  function percent(value, places) {
    return isNumber(value) ? (value * 100).toFixed(places || 0) + "%" : "n/a";
  }

  function count(value) {
    return isNumber(value) ? Math.round(value).toLocaleString() : "n/a";
  }

  /* Compute time, as headroom.report.tables.duration formats it, so the page and the
     README's fit-time column read the same. */
  function duration(seconds) {
    if (!isNumber(seconds)) { return "n/a"; }
    if (seconds < 3600) { return Math.max(1, Math.round(seconds / 60)) + " min"; }
    return (seconds / 3600).toFixed(1) + " h";
  }

  function signedPercent(value, places) {
    if (!isNumber(value)) { return "n/a"; }
    return (value < 0 ? "−" : "+") + Math.abs(value * 100).toFixed(places === undefined ? 1 : places) + "%";
  }

  function roundTo(value, unit) {
    return isNumber(value) ? Math.round(value / unit) * unit : null;
  }

  /* A model's label carries its own comma ("N-HiTS, refitted every 4 weeks"), and two of
     those inside a list make the sentence unreadable, so prose uses the name alone. */
  function shortName(name) { return name.split(", ")[0]; }

  /* What each model is, for a reader who has not met it, keyed by the name prose uses. This
     is description, not measurement: every number about a model is on the chart and in the
     table, from the payload. A model the payload names and this does not is shown as plain
     text rather than with a wrong note, and tests/test_export.py fails until one is written. */
  var MODEL_NOTES = {
    "Seasonal naive": "The baseline every other model is measured against. It forecasts each " +
      "day as the same weekday one or two weeks earlier, and takes its range from the spread " +
      "of its own past errors at that distance ahead. Nothing is fitted, and on demand with a " +
      "strong weekly cycle it is harder to beat than it sounds.",
    "ETS": "Exponential smoothing: a classical statistical model that tracks each series' " +
      "level, trend and weekly pattern, weighting recent days most. It is fitted to each of " +
      "the 37 series separately, with its form chosen automatically, and it is the " +
      "forecaster the top of this page draws.",
    "Theta": "A classical statistical method that takes out the weekly pattern, splits what " +
      "is left into a long-run trend and short-run movement, forecasts each and puts them " +
      "back together. It won the M3 forecasting competition and has been a standard " +
      "benchmark since. Fitted to each series separately.",
    "MSTL": "Splits each series into a weekly pattern, a yearly pattern and what is left, " +
      "forecasts the remainder with exponential smoothing and adds the patterns back. The " +
      "one statistical model here that sees the annual cycle as well as the weekly one.",
    "AutoARIMA": "The classical ARIMA model, which forecasts a series from its own recent " +
      "values and its own recent errors, with the model's orders searched automatically for " +
      "each series at every forecast. That search is why it is by far the most expensive " +
      "statistical model to fit.",
    "LightGBM": "Gradient-boosted decision trees: one model trained across all 37 series at " +
      "once, from each series' recent values and a calendar that includes public holidays. " +
      "It forecasts a single middle value, so its range is conformal, built from its own " +
      "past errors.",
    "N-HiTS": "A deep neural network designed for forecasting, which reads each series at " +
      "several time scales at once and combines what it sees. Trained on these 37 series and " +
      "refitted every four weeks, it forecasts a full set of quantiles directly.",
    "PatchTST": "A transformer, the architecture behind large language models, adapted to " +
      "time series by cutting each series into short patches and reading them the way a " +
      "language model reads words. Trained on these 37 series and refitted every thirteen " +
      "weeks, it forecasts a full set of quantiles directly.",
    "TimesFM": "Google Research's pretrained time-series foundation model, version 2.5 with " +
      "200 million parameters, trained on a large public corpus of time series and used here " +
      "zero-shot: nothing was fitted to this data. Its training data covers the pandemic " +
      "years in other series, so it is judged only on forecasts made after that data ends."
  };

  var noteCount = 0;

  /* A model's name with its description attached: a bubble on hover or focus for a reader
     who can see it, and the same words as a description for one who hears the page. */
  function term(name, notesHost) {
    var note = MODEL_NOTES[name];
    if (!note) { return document.createTextNode(name); }
    var id = "model-note-" + (noteCount++);
    notesHost.appendChild(el("span", { id: id }, note));
    return el("span", {
      "class": "term", tabindex: "0", "data-note": note, "aria-describedby": id
    }, name);
  }

  function termList(names, notesHost) {
    var out = document.createDocumentFragment();
    names.forEach(function (name, i) {
      if (i) { out.appendChild(document.createTextNode(i === names.length - 1 ? " and " : ", ")); }
      out.appendChild(term(name, notesHost));
    });
    return out;
  }

  /* Names, read out the way a sentence reads them. */
  function listing(items) {
    if (!items.length) { return ""; }
    if (items.length === 1) { return items[0]; }
    return items.slice(0, -1).join(", ") + " and " + items[items.length - 1];
  }

  function dateLabel(at) {
    var when = new Date(at * DAY);
    return when.getUTCDate() + " " + MONTHS[when.getUTCMonth()] + " " + when.getUTCFullYear();
  }

  /* An estimate is never shown without its interval: the interval is the measurement.
     A signed number is coloured by which way is good, and that differs by column: a change
     in CRPS is a loss, so less is better, and a skill is the loss removed, so more is. */
  function estimateCell(estimate, places, showSign, upIsBetter) {
    var cell = el("td", { "class": "num" });
    var format = showSign ? signed : decimals;
    var point = el("span", null, format(estimate.point, places));
    if (showSign && isNumber(estimate.point)) {
      var good = upIsBetter ? estimate.point > 0 : estimate.point < 0;
      point.className = good ? "better" : "worse";
    }
    cell.appendChild(point);
    cell.appendChild(document.createTextNode(" "));
    cell.appendChild(el(
      "span",
      { "class": "interval" },
      "[" + format(estimate.low, places) + ", " + format(estimate.high, places) + "]"
    ));
    return cell;
  }

  function legendInto(host, entries) {
    clear(host);
    entries.forEach(function (entry) {
      var item = el("span", { "class": "legend-item" });
      item.appendChild(el("span", { "class": "swatch " + entry.swatch }));
      item.appendChild(document.createTextNode(entry.label));
      host.appendChild(item);
    });
  }

  /* Charts -------------------------------------------------------------------- */

  function ticks(lo, hi, wanted) {
    var span = hi - lo;
    if (!(span > 0)) { return [lo]; }
    var raw = span / wanted;
    var magnitude = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = 10 * magnitude;
    var steps = [1, 2, 2.5, 5, 10];
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] * magnitude >= raw) { step = steps[i] * magnitude; break; }
    }
    var out = [];
    for (var t = Math.ceil(lo / step) * step; t <= hi + step / 1e6; t += step) { out.push(t); }
    return out;
  }

  function timeTicks(lo, hi) {
    var out = [];
    var years = new Date(hi * DAY).getUTCFullYear() - new Date(lo * DAY).getUTCFullYear();
    if (years >= 3) {
      var first = new Date(lo * DAY).getUTCFullYear();
      var last = new Date(hi * DAY).getUTCFullYear();
      var every = Math.max(1, Math.ceil((last - first + 1) / 9));
      for (var year = first; year <= last; year += every) {
        var at = Date.UTC(year, 0, 1) / DAY;
        if (at >= lo && at <= hi) { out.push({ at: at, label: String(year) }); }
      }
    } else {
      var start = new Date(lo * DAY);
      var month = Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1) / DAY;
      var gap = (hi - lo) > 500 ? 3 : ((hi - lo) > 200 ? 2 : 1);
      var index = 0;
      while (month <= hi) {
        if (month >= lo) {
          var when = new Date(month * DAY);
          if (index % gap === 0) {
            out.push({
              at: month,
              label: MONTHS[when.getUTCMonth()] + (when.getUTCMonth() === 0 ? " " + when.getUTCFullYear() : "")
            });
          }
          index++;
        }
        var next = new Date(month * DAY);
        month = Date.UTC(next.getUTCFullYear(), next.getUTCMonth() + 1, 1) / DAY;
      }
    }
    return out;
  }

  function chart(height, xDomain, yDomain, options) {
    var inner = WIDTH - MARGIN.left - MARGIN.right;
    var deep = height - MARGIN.top - MARGIN.bottom;
    var root = svg("svg", {
      viewBox: "0 0 " + WIDTH + " " + height,
      role: "img",
      "aria-label": (options && options.label) || "chart"
    });
    var plot = svg("g", { transform: "translate(" + MARGIN.left + "," + MARGIN.top + ")" });
    root.appendChild(plot);

    var xSpan = xDomain[1] - xDomain[0] || 1;
    var ySpan = yDomain[1] - yDomain[0] || 1;
    var frame = {
      root: root,
      plot: plot,
      inner: inner,
      deep: deep,
      domain: xDomain,
      x: function (value) { return (value - xDomain[0]) / xSpan * inner; },
      y: function (value) { return deep - (value - yDomain[0]) / ySpan * deep; },
      at: function (offset) { return xDomain[0] + offset / inner * xSpan; }
    };

    var axes = svg("g");
    plot.appendChild(axes);
    ticks(yDomain[0], yDomain[1], 4).forEach(function (value) {
      var at = frame.y(value);
      if (at < -1 || at > deep + 1) { return; }
      axes.appendChild(svg("line", { "class": "gridline", x1: 0, x2: inner, y1: at, y2: at }));
      var text = svg("text", { "class": "tick", x: -9, y: at + 4, "text-anchor": "end" });
      text.textContent = (options && options.formatY) ? options.formatY(value) : String(value);
      axes.appendChild(text);
    });
    timeTicks(xDomain[0], xDomain[1]).forEach(function (tick) {
      var at = frame.x(tick.at);
      axes.appendChild(svg("line", { "class": "axis-line", x1: at, x2: at, y1: deep, y2: deep + 4 }));
      var text = svg("text", { "class": "tick", x: at, y: deep + 18, "text-anchor": "middle" });
      text.textContent = tick.label;
      axes.appendChild(text);
    });
    axes.appendChild(svg("line", { "class": "axis-line", x1: 0, x2: inner, y1: deep, y2: deep }));
    if (options && options.yTitle) {
      var title = svg("text", {
        "class": "axis-title",
        transform: "translate(" + (-MARGIN.left + 13) + "," + (deep / 2) + ") rotate(-90)",
        "text-anchor": "middle"
      });
      title.textContent = options.yTitle;
      frame.plot.appendChild(title);
    }
    return frame;
  }

  function shade(frame, from, to, label, low) {
    var lo = Math.max(from, frame.domain[0]);
    var hi = Math.min(to, frame.domain[1]);
    if (hi <= lo) { return; }
    frame.plot.appendChild(svg("rect", {
      "class": "shift-span",
      x: frame.x(lo), y: 0, width: frame.x(hi) - frame.x(lo), height: frame.deep
    }));
    if (label) {
      /* At the top by default, and at the foot where the series live up there: on a
         coverage chart drawn 0 to 100 the lines sit against the ceiling. */
      var text = svg("text", {
        "class": "shift-label", x: frame.x(lo) + 4, y: low ? frame.deep - 5 : 11
      });
      text.textContent = label;
      frame.plot.appendChild(text);
    }
  }

  /* Both of these break the path wherever a value is missing, rather than drawing a straight
     line across a gap as though something had been measured there. */
  function linePath(frame, xs, values) {
    var parts = [];
    var open = false;
    for (var i = 0; i < values.length; i++) {
      if (!isNumber(values[i])) { open = false; continue; }
      parts.push((open ? "L" : "M") + frame.x(xs[i]).toFixed(1) + " " + frame.y(values[i]).toFixed(1));
      open = true;
    }
    return parts.join(" ");
  }

  function areaPath(frame, xs, lower, upper) {
    var parts = [];
    var run = [];
    function flush() {
      if (run.length >= 2) {
        var ahead = run.map(function (i) {
          return frame.x(xs[i]).toFixed(1) + " " + frame.y(lower[i]).toFixed(1);
        });
        var back = [];
        for (var j = run.length - 1; j >= 0; j--) {
          back.push(frame.x(xs[run[j]]).toFixed(1) + " " + frame.y(upper[run[j]]).toFixed(1));
        }
        parts.push("M" + ahead.join("L") + "L" + back.join("L") + "Z");
      }
      run = [];
    }
    for (var i = 0; i < xs.length; i++) {
      if (isNumber(lower[i]) && isNumber(upper[i])) { run.push(i); } else { flush(); }
    }
    flush();
    return parts.join(" ");
  }

  /* Hover. The chart is scaled to the page by its viewBox, so a pointer position has to be
     put back into viewBox units before it means anything. */
  function attachHover(frame, xs, show, hide) {
    var overlay = svg("g");
    frame.plot.appendChild(overlay);
    var catcher = svg("rect", {
      "class": "catcher", x: 0, y: 0, width: frame.inner, height: frame.deep, fill: "transparent"
    });
    frame.plot.appendChild(catcher);

    function nearest(event) {
      var box = frame.root.getBoundingClientRect();
      if (!box.width) { return -1; }
      var offset = (event.clientX - box.left) / box.width * WIDTH - MARGIN.left;
      var wanted = frame.at(Math.max(0, Math.min(frame.inner, offset)));
      var best = 0;
      for (var i = 1; i < xs.length; i++) {
        if (Math.abs(xs[i] - wanted) < Math.abs(xs[best] - wanted)) { best = i; }
      }
      return best;
    }

    function move(event) {
      var index = nearest(event);
      if (index < 0) { return; }
      clear(overlay);
      overlay.appendChild(svg("line", {
        "class": "cursor", x1: frame.x(xs[index]), x2: frame.x(xs[index]), y1: 0, y2: frame.deep
      }));
      show(index, overlay);
    }

    catcher.addEventListener("pointermove", move);
    catcher.addEventListener("pointerdown", move);
    catcher.addEventListener("pointerleave", function () { clear(overlay); hide(); });
    return overlay;
  }

  function dot(frame, x, y, className) {
    return svg("circle", { "class": "dot " + (className || ""), cx: frame.x(x), cy: frame.y(y), r: 3.5 });
  }

  /* The rolling mean, as headroom.report.charts.rolling_mean computes it: trailing, and only
     where the whole window was measured. A window that is part gap is not an average of
     anything. */
  function rolling(values, width) {
    var out = new Array(values.length);
    for (var i = 0; i < values.length; i++) { out[i] = null; }
    for (var end = width - 1; end < values.length; end++) {
      var total = 0;
      var whole = true;
      for (var j = end - width + 1; j <= end; j++) {
        if (!isNumber(values[j])) { whole = false; break; }
        total += values[j];
      }
      if (whole) { out[end] = total / width; }
    }
    return out;
  }

  function mean(values) {
    var total = 0;
    var n = 0;
    values.forEach(function (value) { if (isNumber(value)) { total += value; n++; } });
    return n ? total / n : null;
  }

  function lowest(values) {
    var least = null;
    values.forEach(function (value) {
      if (isNumber(value) && (least === null || value < least)) { least = value; }
    });
    return least;
  }

  function extent(serieses, pad) {
    var lo = Infinity;
    var hi = -Infinity;
    serieses.forEach(function (values) {
      values.forEach(function (value) {
        if (!isNumber(value)) { return; }
        if (value < lo) { lo = value; }
        if (value > hi) { hi = value; }
      });
    });
    if (lo === Infinity) { return [0, 1]; }
    var room = (hi - lo) * (pad === undefined ? 0.08 : pad) || 1;
    return [lo - room, hi + room];
  }

  /* A cost here is counted in unit-days: a unit staffed and not needed costs 1, and one
     needed and missing costs 4 of the same thing. So a cost saved is a number of unit-days
     saved, and the only input that turns it into a quantity a planner recognises is the
     hours in a unit-day, which lives in inputs/decision.toml and not in this file. Nothing
     here invents a rate of pay: a reader with one can put it on the crew-hours. */
  var YEAR = 365;

  function worth(saved, hoursPerUnitDay) {
    if (!isNumber(saved) || !isNumber(hoursPerUnitDay) || saved <= 0) { return ""; }
    var days = roundTo(saved * YEAR, 100);
    var hours = roundTo(saved * YEAR * hoursPerUnitDay, 1000);
    return "about " + count(days) + " crew-days a year, or " + count(hours) +
      " crew-hours at " + decimals(hoursPerUnitDay, 0) + " hours to a unit-day";
  }

  /* The forecast panel -------------------------------------------------------- */

  function serviceQuantile(bands, levels, wanted, at) {
    var lo = 0;
    while (lo < levels.length - 2 && levels[lo + 1] < wanted) { lo++; }
    var hi = lo + 1;
    var low = bands[lo][at];
    var high = bands[hi][at];
    if (!isNumber(low) || !isNumber(high)) { return null; }
    var span = levels[hi] - levels[lo];
    var held = Math.min(Math.max(wanted, levels[0]), levels[levels.length - 1]);
    return low + (high - low) * (span > 0 ? (held - levels[lo]) / span : 0);
  }

  function fanWindow(days) {
    if (state.range === "all") { return [0, days.length - 1]; }
    var from;
    var to;
    if (state.range === "shift") {
      from = day("2019-09-01");
      to = day("2020-09-01");
    } else {
      to = day(days[days.length - 1]);
      from = to - 365;
    }
    var first = 0;
    var last = days.length - 1;
    while (first < last && day(days[first]) < from) { first++; }
    while (last > first && day(days[last]) > to) { last--; }
    return [first, last];
  }

  function drawFan() {
    var payload = state.forecast;
    var host = byId("fan-chart");
    clear(host);
    if (!payload) { return; }

    var span = fanWindow(payload.days);
    var xs = [];
    var labels = [];
    for (var i = span[0]; i <= span[1]; i++) {
      xs.push(day(payload.days[i]));
      labels.push(payload.days[i]);
    }
    var node = payload.bands[state.node].map(function (level) {
      return level.slice(span[0], span[1] + 1);
    });
    var actual = payload.actual[state.node].slice(span[0], span[1] + 1);
    var levels = payload.levels;
    var middle = Math.floor(levels.length / 2);
    var service = state.dashboard.staffing.service_levels[state.service];
    var serviceLine = xs.map(function (ignored, k) {
      return serviceQuantile(node, levels, service, k);
    });

    /* Seventeen years at this width is 911 origins in 806 pixels, and a band drawn over
       that many points is a smear rather than a range: the two lines are the only things
       still legible, so they are the only things drawn, thinner, and the caption says the
       band is still there to be read at a shorter period or off the hover. */
    var dense = state.range === "all";
    var thin = dense ? " thin" : "";
    var frame = chart(
      dense ? 300 : 320,
      [xs[0], xs[xs.length - 1]],
      extent(dense ? [actual, node[middle]] : node.concat([actual])),
      { label: "Forecast bands against what happened", formatY: count, yTitle: "incidents a day" }
    );
    shade(frame, day("2020-03-01"), day("2020-06-01"), "the shift");

    if (!dense) {
      frame.plot.appendChild(svg("path", {
        "class": "band", d: areaPath(frame, xs, node[0], node[levels.length - 1]), "fill-opacity": 0.15
      }));
      if (levels.length >= 4) {
        frame.plot.appendChild(svg("path", {
          "class": "band", d: areaPath(frame, xs, node[1], node[levels.length - 2]), "fill-opacity": 0.22
        }));
      }
    }
    frame.plot.appendChild(svg("path", { "class": "median" + thin, d: linePath(frame, xs, node[middle]) }));
    if (!dense) {
      frame.plot.appendChild(svg("path", { "class": "service", d: linePath(frame, xs, serviceLine) }));
    }
    frame.plot.appendChild(svg("path", { "class": "actual-line" + thin, d: linePath(frame, xs, actual) }));

    var readout = byId("fan-readout");
    attachHover(frame, xs, function (index, overlay) {
      if (isNumber(actual[index])) {
        overlay.appendChild(dot(frame, xs[index], actual[index], "actual"));
      }
      if (isNumber(node[middle][index])) {
        overlay.appendChild(dot(frame, xs[index], node[middle][index], "median"));
      }
      clear(readout);
      readout.appendChild(el("span", { "class": "when" }, dateLabel(xs[index])));
      [
        ["actual", count(actual[index])],
        ["forecast", count(node[middle][index])],
        [percent(levels[0]) + " to " + percent(levels[levels.length - 1]),
          count(node[0][index]) + " to " + count(node[levels.length - 1][index])],
        ["staff to", count(serviceLine[index])]
      ].forEach(function (pair) {
        var slot = el("span", { "class": "pair" });
        slot.appendChild(el("span", { "class": "key" }, pair[0] + " "));
        slot.appendChild(document.createTextNode(pair[1]));
        readout.appendChild(slot);
      });
    }, function () {
      readout.textContent = "Point at the chart to read a day off it.";
    });
    readout.textContent = "Point at the chart to read a day off it.";
    host.appendChild(frame.root);

    var entries = [
      { label: "what happened", swatch: "sw-ink" },
      { label: "the forecast (median)", swatch: "sw-accent" }
    ];
    if (!dense) {
      entries.push({
        label: percent(levels[1]) + " to " + percent(levels[levels.length - 2]) + " and "
          + percent(levels[0]) + " to " + percent(levels[levels.length - 1]),
        swatch: "sw-band block"
      });
      entries.push({ label: "staffed to " + percent(service), swatch: "sw-warn dashed" });
    }
    legendInto(byId("fan-legend"), entries);

    byId("fan-caption").textContent =
      payload.model + " forecasting " + payload.horizon_step + " days ahead for " +
      payload.nodes[state.node] + ", " + labels[0] + " to " + labels[labels.length - 1] +
      ". Each point is one weekly forecast read at the same distance ahead, so the line is a " +
      "record of the same task repeated, not one forecast extended." +
      (dense
        ? " Over every origin the band and the staffing line are left off and the lines are"
          + " drawn thin, because 911 forecasts of bands overlap into a block of colour."
          + " Pick a shorter period to see them, or read them off the hover, which still"
          + " carries them."
        : "");
  }

  /* The coverage panel -------------------------------------------------------- */

  function drawCoverage() {
    var section = state.dashboard.coverage;
    var host = byId("coverage-chart");
    clear(host);
    var xs = section.days.map(day);
    var chosen = section.series.filter(function (one) {
      return Math.abs(one.nominal - state.nominal) < 1e-9;
    });
    var covers = chosen.map(function (one) { return rolling(one.coverage, state.window); });
    var widths = chosen.map(function (one) { return rolling(one.width, state.window); });
    var domain = [xs[0], xs[xs.length - 1]];

    /* Fixed from nothing to everything, not fitted to the window on screen. A scale that
       grows out of the data makes a collapse to 20 percent look the same size as a wobble
       of two points, and this chart exists to show the difference between them. */
    var top = chart(260, domain, [0, 1], {
      label: "Rolling coverage against nominal",
      formatY: function (value) { return (value * 100).toFixed(0) + "%"; },
      yTitle: "outcomes inside the interval"
    });
    shade(top, day(section.shift[0]), day(section.shift[1]), "the shift", true);
    top.plot.appendChild(svg("line", {
      "class": "nominal", x1: 0, x2: top.inner, y1: top.y(state.nominal), y2: top.y(state.nominal)
    }));
    covers.forEach(function (values, i) {
      top.plot.appendChild(svg("path", { "class": "series m" + (i % 3), d: linePath(top, xs, values) }));
    });

    var readout = byId("coverage-readout");
    attachHover(top, xs, function (index, overlay) {
      clear(readout);
      readout.appendChild(el("span", { "class": "when" }, dateLabel(xs[index])));
      covers.forEach(function (values, i) {
        if (isNumber(values[index])) {
          overlay.appendChild(dot(top, xs[index], values[index], "m" + (i % 3)));
        }
        var slot = el("span", { "class": "pair" });
        slot.appendChild(el("span", { "class": "key" }, chosen[i].method + " "));
        slot.appendChild(document.createTextNode(percent(values[index], 1)));
        readout.appendChild(slot);
      });
    }, function () {
      readout.textContent = "Point at the chart to read a window off it.";
    });
    readout.textContent = "Point at the chart to read a window off it.";
    host.appendChild(top.root);

    var bottom = chart(225, domain, extent(widths), {
      label: "Mean interval width",
      formatY: count,
      yTitle: "width, incidents"
    });
    shade(bottom, day(section.shift[0]), day(section.shift[1]), null);
    widths.forEach(function (values, i) {
      bottom.plot.appendChild(svg("path", {
        "class": "series m" + (i % 3), d: linePath(bottom, xs, values)
      }));
    });
    host.appendChild(bottom.root);

    legendInto(byId("coverage-legend"), chosen.map(function (one, i) {
      return { label: one.method, swatch: "sw" + (i % 3) };
    }).concat([{ label: "what was promised", swatch: "sw-faint dashed" }]));

    byId("coverage-caption").textContent =
      "Above: the share of outcomes that fell inside the " + percent(state.nominal) +
      " interval, in a trailing window of " + state.window + " weekly origins, at the " +
      section.level + " level. Below: the mean width of those same intervals. Both are drawn " +
      "from the same forecasts, so a method that buys coverage by widening shows it here. " +
      "The coverage scale is fixed from 0 to 100 percent whatever the window, so the fall " +
      "is the same size every time you look at it.";
  }

  /* The models panel ---------------------------------------------------------- */

  /* Thin bars: the length is the measurement and the ink is not. Three to a model, one per
     level of the hierarchy, in one hue getting lighter as the series get smaller, because
     the three levels are an ordered set and not three unrelated things. */
  var BAR = 13;
  var BAR_GAP = 3;
  var GROUP_GAP = 18;
  var OWN_GAP = 34;
  var MODELS_MARGIN = { top: 12, right: 68, bottom: 34, left: 190 };

  /* Decimals that show three significant figures, as headroom.report.tables.decimals_for
     chooses them, so a CRPS of 135 and one of 8.27 both read as measurements. */
  function sigPlaces(value) {
    if (!isNumber(value)) { return 2; }
    var size = Math.abs(value);
    if (size >= 100) { return 0; }
    if (size >= 10) { return 1; }
    return 2;
  }

  function withInterval(estimate, places) {
    return decimals(estimate.point, places) + " [" + decimals(estimate.low, places) +
      ", " + decimals(estimate.high, places) + "]";
  }

  /* What the chart draws: every model except the baseline, which is the zero line itself,
     and then any model scored on its own window, under a divider. */
  function modelGroups(section) {
    var groups = [];
    section.rows.forEach(function (row) {
      if (row.name !== section.baseline) { groups.push({ row: row, own: false }); }
    });
    section.own_windows.forEach(function (row) { groups.push({ row: row, own: true }); });
    return groups;
  }

  function skillExtent(groups) {
    var lo = 0;
    var hi = 0;
    groups.forEach(function (group) {
      group.row.levels.forEach(function (one) {
        if (!one.skill) { return; }
        [one.skill.low, one.skill.point, one.skill.high].forEach(function (value) {
          if (!isNumber(value)) { return; }
          if (value < lo) { lo = value; }
          if (value > hi) { hi = value; }
        });
      });
    });
    var room = (hi - lo) * 0.04 || 0.01;
    return [lo - room, hi + room];
  }

  function modelsTable(section) {
    var table = byId("models-table");
    var caption = table.querySelector("caption");
    clear(table);
    table.appendChild(caption);

    var head = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", { scope: "col" }, "Model"));
    headRow.appendChild(el("th", { scope: "col" }, "Level"));
    ["CRPS", "CRPS skill", "Coverage at " + percent(section.nominal), "Mean width"]
      .forEach(function (name) {
        headRow.appendChild(el("th", { scope: "col", "class": "num" }, name));
      });
    headRow.appendChild(el("th", { scope: "col", "class": "num" }, "Compute"));
    head.appendChild(headRow);
    table.appendChild(head);

    var body = el("tbody");
    section.rows.concat(section.own_windows).forEach(function (row) {
      row.levels.forEach(function (one, j) {
        var line = el("tr");
        if (row.name === section.baseline) { line.className = "reference"; }
        var cell = el("th", { scope: "row", "class": row.window ? "wrap" : "" });
        if (j === 0) {
          cell.appendChild(document.createTextNode(row.name));
          if (row.window) {
            cell.appendChild(el(
              "span",
              { "class": "interval" },
              " " + row.origins + " origins from " + row.first
            ));
          }
        }
        line.appendChild(cell);
        line.appendChild(el("td", null, one.level));
        line.appendChild(estimateCell(one.crps, sigPlaces(one.crps.point), false));
        if (one.skill) {
          line.appendChild(estimateCell(one.skill, 3, true, true));
        } else {
          line.appendChild(el("td", { "class": "num" }, "reference"));
        }
        line.appendChild(estimateCell(one.coverage, 3, false));
        line.appendChild(estimateCell(one.width, sigPlaces(one.width.point), false));
        line.appendChild(el("td", { "class": "num" }, j === 0 ? duration(row.fit_seconds) : ""));
        body.appendChild(line);
      });
    });
    table.appendChild(body);

    byId("models-table-caption").textContent =
      "The same numbers the chart draws, and the two it cannot: what each model's intervals " +
      "covered at the " + percent(section.nominal) + " nominal level, and how wide they were. " +
      "A skill is signed because it is a difference, and a negative one is worse than " +
      section.baseline + ". Every estimate carries its 95 percent moving-block bootstrap " +
      "interval over forecast origins. A skill is shown here as a share, and on the chart as " +
      "a percentage of the same thing.";
  }

  function drawModels() {
    var section = state.dashboard.models;
    var host = byId("models-chart");
    clear(host);
    var groups = modelGroups(section);
    if (!groups.length) { return; }

    var names = groups[0].row.levels.map(function (one) { return one.level; });
    var span = skillExtent(groups);
    var inner = WIDTH - MODELS_MARGIN.left - MODELS_MARGIN.right;
    var groupDeep = names.length * (BAR + BAR_GAP) - BAR_GAP;
    var tops = [];
    var deep = 0;
    groups.forEach(function (group, i) {
      if (i) { deep += (group.own && !groups[i - 1].own) ? OWN_GAP : GROUP_GAP; }
      tops.push(deep);
      deep += groupDeep;
    });
    var height = deep + MODELS_MARGIN.top + MODELS_MARGIN.bottom;

    var root = svg("svg", {
      viewBox: "0 0 " + WIDTH + " " + height,
      role: "img",
      "aria-label": "CRPS skill against " + section.baseline + ", every model, three levels"
    });
    var plot = svg("g", {
      transform: "translate(" + MODELS_MARGIN.left + "," + MODELS_MARGIN.top + ")"
    });
    root.appendChild(plot);
    function x(value) { return (value - span[0]) / (span[1] - span[0]) * inner; }

    ticks(span[0], span[1], 5).forEach(function (value) {
      var at = x(value);
      if (at < -1 || at > inner + 1) { return; }
      plot.appendChild(svg("line", { "class": "gridline", x1: at, x2: at, y1: -6, y2: deep }));
      var tick = svg("text", { "class": "tick", x: at, y: deep + 17, "text-anchor": "middle" });
      tick.textContent = (value * 100).toFixed(0) + "%";
      plot.appendChild(tick);
    });
    plot.appendChild(svg("line", { "class": "zero", x1: x(0), x2: x(0), y1: -6, y2: deep }));
    var zero = svg("text", {
      "class": "zero-label", x: x(0), y: deep + 31, "text-anchor": "middle"
    });
    zero.textContent = section.baseline + ", the baseline";
    plot.appendChild(zero);

    var readout = byId("models-readout");
    function reset() { readout.textContent = "Point at a bar to read that model off it."; }
    function show(row, one, own) {
      clear(readout);
      readout.appendChild(el("span", { "class": "when" }, row.name + ", " + one.level));
      [
        ["CRPS", withInterval(one.crps, sigPlaces(one.crps.point))],
        ["skill", one.skill ? signedPercent(one.skill.point, 1) : "reference"],
        ["coverage at " + percent(section.nominal), percent(one.coverage.point, 1)],
        ["width", count(one.width.point)],
        [own ? "to run" : "to fit", duration(row.fit_seconds)]
      ].forEach(function (pair) {
        var slot = el("span", { "class": "pair" });
        slot.appendChild(el("span", { "class": "key" }, pair[0] + " "));
        slot.appendChild(document.createTextNode(pair[1]));
        readout.appendChild(slot);
      });
    }

    groups.forEach(function (group, i) {
      var row = group.row;
      var top = tops[i];
      if (group.own && (i === 0 || !groups[i - 1].own)) {
        var at = top - OWN_GAP / 2;
        plot.appendChild(svg("line", {
          "class": "divider", x1: 8 - MODELS_MARGIN.left, x2: inner, y1: at, y2: at
        }));
        var note = svg("text", {
          "class": "divider-label", x: 8 - MODELS_MARGIN.left, y: at - 7
        });
        note.textContent = "pretrained, on its own window: never pooled with the models above";
        plot.appendChild(note);
      }

      var parts = row.name.split(", ");
      var lines = [
        parts[0],
        parts.slice(1).join(", "),
        (group.own ? row.origins + " origins, " : "") + duration(row.fit_seconds) +
          (group.own ? " to run" : " to fit")
      ].filter(function (text) { return text; });
      lines.forEach(function (text, j) {
        var label = svg("text", {
          "class": j === 0 ? "row-label" : "row-note",
          x: -13,
          y: top + 11 + j * 13,
          "text-anchor": "end"
        });
        label.textContent = text;
        plot.appendChild(label);
      });

      row.levels.forEach(function (one, j) {
        var y = top + j * (BAR + BAR_GAP);
        /* The hit target is the whole row rather than the bar, because a bar at one percent
           skill is two pixels wide and nobody can point at it. */
        var catcher = svg("rect", {
          "class": "catcher",
          x: 8 - MODELS_MARGIN.left,
          y: y - BAR_GAP / 2,
          width: inner + MODELS_MARGIN.left - 8,
          height: BAR + BAR_GAP,
          fill: "transparent"
        });
        plot.appendChild(catcher);
        catcher.addEventListener("pointerenter", function () { show(row, one, group.own); });
        catcher.addEventListener("pointerdown", function () { show(row, one, group.own); });

        var skill = one.skill;
        if (!skill || !isNumber(skill.point)) { return; }
        var from = x(Math.min(0, skill.point));
        var to = x(Math.max(0, skill.point));
        plot.appendChild(svg("rect", {
          "class": "bar lv" + j + (group.own ? " own" : ""),
          x: from, y: y, width: Math.max(1.5, to - from), height: BAR, rx: 3
        }));
        var end = to;
        if (isNumber(skill.low) && isNumber(skill.high)) {
          var mid = y + BAR / 2;
          plot.appendChild(svg("line", {
            "class": "whisker", x1: x(skill.low), x2: x(skill.high), y1: mid, y2: mid
          }));
          [skill.low, skill.high].forEach(function (edge) {
            plot.appendChild(svg("line", {
              "class": "whisker", x1: x(edge), x2: x(edge), y1: y + 2.5, y2: y + BAR - 2.5
            }));
          });
          end = Math.max(end, x(skill.high));
        }
        var value = svg("text", { "class": "bar-value", x: end + 8, y: y + BAR - 2.5 });
        value.textContent = signedPercent(skill.point, 1);
        plot.appendChild(value);
      });
    });

    root.addEventListener("pointerleave", reset);
    reset();
    host.appendChild(root);

    legendInto(byId("models-legend"), names.map(function (name, j) {
      return { label: name, swatch: "sw-lv" + j + " solid" };
    }));

    var trained = section.rows.map(function (row) { return shortName(row.name); });
    byId("models-lede").textContent =
      "Every model this project trained, on one chart, scored on the same " +
      section.origins.toLocaleString() + " forecasts of the same days: " + listing(trained) +
      ". The bar is CRPS skill: the share of " + section.baseline + "'s loss the model takes " +
      "away, so higher is better and zero is no better than planning from last week. CRPS " +
      "scores the whole distribution rather than the middle of it, which is why it is the bar " +
      "and the average error is not here at all. What each model cost is under its name, " +
      "because two of these bars are the same length at very different prices.";

    var ownSentence = "";
    if (section.own_windows.length) {
      var shot = section.own_windows[0];
      ownSentence = " " + shortName(shot.name) + " sits under the divider rather than among the bars " +
        "above it: it is pretrained, its weights postdate most of these origins, and it is " +
        "scored only on the window whose whole fortnight falls after its training data (" +
        shot.origins + " origins from " + shot.first + "). Its bars are comparable with each " +
        "other and with nothing above the line.";
    }
    byId("models-caption").textContent =
      "CRPS skill against " + section.baseline + " at each level of the hierarchy: the city, " +
      "the 5 boroughs and the 31 dispatch areas. Whiskers are 95 percent moving-block " +
      "bootstrap intervals over forecast origins, and the comparison is paired, because every " +
      "model forecast the same origins." + ownSentence;

    var conformal = [];
    var theirs = [];
    section.rows.concat(section.own_windows).forEach(function (row) {
      if (row.name === section.baseline) { return; }
      (row.kind === "boosting" || row.kind === "zero-shot" ? conformal : theirs)
        .push(shortName(row.name));
    });
    byId("models-note").textContent =
      "The coverage column is not one promise repeated. " + listing(theirs) + " are scored on " +
      "the quantiles they produce themselves, with no conformal step, so nothing is promised " +
      "for them and the number is only what happened. " + listing(conformal) + " forecast a " +
      "median and are given a conformal distribution built from their own past errors, whose " +
      "guarantee holds on average over time and only where errors are exchangeable, which " +
      "demand through a pandemic is not.";

    modelsTable(section);
  }

  /* The reconciliation panel -------------------------------------------------- */

  function drawReconciliation() {
    var section = state.dashboard.reconciliation;
    var table = byId("reconciliation-table");
    var caption = table.querySelector("caption");
    clear(table);
    table.appendChild(caption);

    var variants = section.rows[0].variants.map(function (one) { return one.name; });
    var head = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", { scope: "col" }, "Level"));
    headRow.appendChild(el("th", { scope: "col", "class": "num" }, "CRPS, " + section.model));
    variants.forEach(function (name) {
      headRow.appendChild(el("th", { scope: "col", "class": "num" }, name + ": change in CRPS"));
      headRow.appendChild(el("th", { scope: "col", "class": "num" },
        name + ": coverage at " + percent(section.nominal)));
    });
    head.appendChild(headRow);
    table.appendChild(head);

    var body = el("tbody");
    section.rows.forEach(function (row) {
      var line = el("tr");
      line.appendChild(el("th", { scope: "row" }, row.level));
      line.appendChild(estimateCell(row.crps, 3, false));
      row.variants.forEach(function (variant) {
        line.appendChild(estimateCell(variant.crps_change, 3, true));
        line.appendChild(estimateCell(variant.coverage, 3, false));
      });
      body.appendChild(line);
    });
    table.appendChild(body);

    byId("reconciliation-lede").textContent =
      "Forecast each area, each borough and the city separately and they will not agree: the " +
      "parts here miss their total by up to " + decimals(section.base_coherence_error, 1) +
      " incidents, which is a plan that cannot be carried out, since the crews assigned to the " +
      "boroughs are the crews the city has. MinT finds the nearest set of forecasts that do " +
      "add up. The largest disagreement left afterwards is " +
      section.reconciled_coherence_error.toExponential(1) + ", which is rounding.";

    byId("reconciliation-note").textContent =
      "MinT moves the middle of each forecast and carries its range along. MinT paths moves the " +
      "whole range: every possible future is put through the same correction, so every one of " +
      "them adds up across all 37 series at once, which is what a decision taken over the whole " +
      "hierarchy needs. Its ranges are not supposed to add up, and they do not: the boroughs do " +
      "not have their bad days together, so the city's busiest plausible day is quieter than the " +
      "sum of theirs. Nothing is floored at zero, because that would break the very thing this " +
      "buys: " + percent(section.negative_share, 3) + " of values fall below it.";
  }

  /* The staffing panel -------------------------------------------------------- */

  function statCard(figure, caption, range, tone) {
    var card = el("div", { "class": "stat" + (tone ? " " + tone : "") });
    card.appendChild(el("span", { "class": "figure" }, figure));
    if (range) { card.appendChild(el("span", { "class": "range" }, range)); }
    card.appendChild(el("span", { "class": "caption" }, caption));
    return card;
  }

  function drawStaffing() {
    var section = state.dashboard.staffing;
    var at = state.service;
    var level = section.service_levels[at];
    var table = byId("staffing-table");
    var caption = table.querySelector("caption");
    clear(table);
    table.appendChild(caption);

    var head = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", { scope: "col" }, "Method"));
    ["Units staffed a day", "Cost a day against the oracle", "Cost against " + section.reference]
      .forEach(function (name) {
        headRow.appendChild(el("th", { scope: "col", "class": "num" }, name));
      });
    head.appendChild(headRow);
    table.appendChild(head);

    var body = el("tbody");
    var reference = null;
    var naive = null;
    section.methods.forEach(function (method) {
      var line = el("tr");
      if (method.name === section.reference) {
        line.className = "reference";
        reference = method;
      }
      if (method.name === "Seasonal naive") { naive = method; }
      line.appendChild(el("th", { scope: "row" }, method.name));
      line.appendChild(estimateCell(method.units[at], 1, false));
      line.appendChild(estimateCell(method.cost[at], 2, false));
      var change = method.cost_change[at];
      line.appendChild(change ? estimateCell(change, 2, true)
        : el("td", { "class": "num" }, "reference"));
      body.appendChild(line);
    });
    table.appendChild(body);

    var headline = byId("staffing-headline");
    clear(headline);
    if (reference) {
      headline.appendChild(statCard(
        decimals(reference.units[at].point, 1),
        "units a day, " + section.reference + " at this service level",
        "[" + decimals(reference.units[at].low, 1) + ", " + decimals(reference.units[at].high, 1) + "]"
      ));
      headline.appendChild(statCard(
        decimals(reference.cost[at].point, 2),
        "cost a day of being wrong, against an oracle that knew demand",
        "[" + decimals(reference.cost[at].low, 2) + ", " + decimals(reference.cost[at].high, 2) + "]"
      ));
      if (naive) {
        var saved = naive.cost[at].point - reference.cost[at].point;
        var share = naive.cost[at].point > 0 ? saved / naive.cost[at].point : 0;
        headline.appendChild(statCard(
          percent(share, 0) + " lower",
          "than staffing from last week's demand, which costs " +
            decimals(naive.cost[at].point, 2) + " a day here",
          null,
          saved > 0 ? "up" : "down"
        ));
        headline.appendChild(statCard(
          decimals(saved, 1) + " unit-days",
          "of idle or missing cover avoided a day: " +
            worth(saved, section.inputs.hours_per_unit_day),
          null,
          saved > 0 ? "up" : "down"
        ));
      }
    }

    var implied = section.inputs.implied_service_level;
    byId("service-level-out").textContent =
      percent(level) + (Math.abs(level - implied) < 1e-9 ? ", implied by the costs" : "");
    byId("staffing-lede").textContent =
      "Staff too few crews and calls wait; staff too many and the money is gone either way. " +
      "One unit here serves " + section.inputs.demand_per_unit + " incidents a day, a spare " +
      "unit-day costs " + section.inputs.cost_over + " and a missing one " +
      section.inputs.cost_under + ", so the arithmetic says to staff to the " + percent(implied) +
      " busiest day the forecast thinks is plausible. Those three numbers are a small table in " +
      "the repository, not constants in the code, and so is the fourth, the " +
      decimals(section.inputs.hours_per_unit_day, 0) + " hours a unit-day stands for, which " +
      "prices nothing and only lets a cost be read as crew-hours. An oracle that knew each " +
      "day's demand would staff " + decimals(section.oracle_units, 1) +
      " units a day and cost nothing.";
  }

  /* Controls ------------------------------------------------------------------ */

  function fillNodes(payload) {
    var select = byId("fan-node");
    var groups = {};
    var titles = { city: "The city", borough: "Boroughs", area: "Dispatch areas" };
    payload.nodes.forEach(function (name, index) {
      var level = payload.node_levels[index];
      if (!groups[level]) {
        groups[level] = el("optgroup", { label: titles[level] || level });
        select.appendChild(groups[level]);
      }
      var parts = name.split("/");
      groups[level].appendChild(el("option", { value: String(index) }, parts[parts.length - 1]));
    });
    select.value = String(state.node);
    select.addEventListener("change", function () {
      state.node = Number(select.value);
      drawFan();
    });
    byId("fan-range").addEventListener("change", function (event) {
      state.range = event.target.value;
      drawFan();
    });
  }

  function fillCoverageControls(section) {
    var nominals = [];
    section.series.forEach(function (one) {
      if (nominals.indexOf(one.nominal) < 0) { nominals.push(one.nominal); }
    });
    nominals.sort(function (a, b) { return a - b; });
    state.nominal = nominals.indexOf(0.9) >= 0 ? 0.9 : nominals[nominals.length - 1];
    var select = byId("coverage-nominal");
    nominals.forEach(function (nominal) {
      var option = el("option", { value: String(nominal) }, percent(nominal) + " interval");
      if (nominal === state.nominal) { option.setAttribute("selected", "selected"); }
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      state.nominal = Number(select.value);
      drawCoverage();
    });

    var slider = byId("coverage-window");
    var out = byId("coverage-window-out");
    state.window = section.window_origins;
    slider.value = String(state.window);
    out.textContent = state.window + " weeks";
    slider.addEventListener("input", function () {
      state.window = Number(slider.value);
      out.textContent = state.window + " weeks";
      drawCoverage();
    });
  }

  function fillServiceControl(section) {
    var slider = byId("service-level");
    var implied = section.service_levels.indexOf(section.inputs.implied_service_level);
    state.service = implied >= 0 ? implied : Math.floor(section.service_levels.length / 2);
    slider.max = String(section.service_levels.length - 1);
    slider.value = String(state.service);
    slider.addEventListener("input", function () {
      state.service = Number(slider.value);
      drawStaffing();
      drawFan();
    });
  }

  function fillHero(payload) {
    var section = payload.coverage;
    var main = section.series.filter(function (one) {
      return Math.abs(one.nominal - 0.9) < 1e-9;
    });
    var chosen = main.length ? main[0] : section.series[0];
    var whole = mean(chosen.coverage);
    var worst = lowest(rolling(chosen.coverage, section.window_origins));
    byId("hero-coverage").textContent = percent(whole, 1);
    byId("hero-coverage-unit").textContent =
      "of outcomes inside the " + percent(chosen.nominal) + " interval (" + chosen.method +
      " conformal), over " + payload.origins.count.toLocaleString() + " forecasts";
    byId("hero-worst").textContent = percent(worst, 1);
    byId("hero-worst-unit").textContent =
      "in the worst " + section.window_origins + " weeks of the seventeen years";

    var staffing = payload.staffing;
    var at = staffing.service_levels.indexOf(staffing.inputs.implied_service_level);
    if (at < 0) { at = 0; }
    var best = null;
    var naive = null;
    staffing.methods.forEach(function (method) {
      if (method.name === staffing.reference) { best = method; }
      if (method.name === "Seasonal naive") { naive = method; }
    });
    if (best && naive && naive.cost[at].point > 0) {
      var saved = naive.cost[at].point - best.cost[at].point;
      var share = saved / naive.cost[at].point;
      byId("hero-saving").textContent = percent(share, 0) + " lower";
      byId("hero-saving-unit").textContent =
        "cost of being wrong, at the service level the costs imply";
      byId("hero-saving-worth").textContent = worth(saved, staffing.inputs.hours_per_unit_day);
    }

    var models = payload.models;
    var trained = models.rows.map(function (row) { return shortName(row.name); });
    var pretrained = models.own_windows.map(function (row) { return shortName(row.name); });
    var line = byId("hero-models");
    var notes = byId("model-notes");
    clear(line);
    clear(notes);
    function words(text) { line.appendChild(document.createTextNode(text)); }
    words("Trained and scored on every one of those forecasts: ");
    line.appendChild(termList(trained, notes));
    words(".");
    if (pretrained.length) {
      words(" ");
      line.appendChild(termList(pretrained, notes));
      words(" is pretrained, trained on none of this data, and is scored on its own clean window.");
    }
    words(" Point at a name to read what it is; they are all on one chart further down.");
    byId("hero-context").textContent =
      payload.origins.count.toLocaleString() + " weekly forecasts of the next " +
      payload.origins.horizon_days + " days, " + payload.origins.first + " to " +
      payload.origins.last + ", over " + payload.hierarchy.nodes.length +
      " series. Nothing is modelled in your browser.";
  }

  /* Start --------------------------------------------------------------------- */

  function fail(message) {
    var box = byId("failure");
    box.textContent = message;
    box.hidden = false;
  }

  function load(name) {
    return fetch("data/" + name).then(function (response) {
      if (!response.ok) { throw new Error(name + ": HTTP " + response.status); }
      return response.json();
    });
  }

  function start() {
    load("dashboard.json").then(function (payload) {
      if (payload.schema_version !== 2) {
        throw new Error("dashboard.json is schema version " + payload.schema_version +
          ", and this page reads version 2");
      }
      state.dashboard = payload;
      fillHero(payload);
      fillCoverageControls(payload.coverage);
      fillServiceControl(payload.staffing);
      drawCoverage();
      drawModels();
      drawReconciliation();
      drawStaffing();
      return load("forecast.json");
    }).then(function (payload) {
      state.forecast = payload;
      byId("fan-step").textContent = String(payload.horizon_step);
      byId("fan-model").textContent = payload.model;
      fillNodes(payload);
      drawFan();
    }).catch(function (error) {
      fail("Could not draw this page: " + error.message +
        ". The same numbers are in the repository's README, as tables.");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
}());
