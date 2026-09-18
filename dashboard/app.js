/* The dashboard, drawn from two data files and nothing else.
 *
 * No framework and no charting library: the page draws four panels, and a library big enough
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

  function dateLabel(at) {
    var when = new Date(at * DAY);
    return when.getUTCDate() + " " + MONTHS[when.getUTCMonth()] + " " + when.getUTCFullYear();
  }

  /* An estimate is never shown without its interval: the interval is the measurement. */
  function estimateCell(estimate, places, showSign) {
    var cell = el("td", { "class": "num" });
    var format = showSign ? signed : decimals;
    var point = el("span", null, format(estimate.point, places));
    if (showSign && isNumber(estimate.point)) {
      point.className = estimate.point < 0 ? "better" : "worse";
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

  function shade(frame, from, to, label) {
    var lo = Math.max(from, frame.domain[0]);
    var hi = Math.min(to, frame.domain[1]);
    if (hi <= lo) { return; }
    frame.plot.appendChild(svg("rect", {
      "class": "shift-span",
      x: frame.x(lo), y: 0, width: frame.x(hi) - frame.x(lo), height: frame.deep
    }));
    if (label) {
      var text = svg("text", { "class": "shift-label", x: frame.x(lo) + 4, y: 11 });
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
      from = to - 730;
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

    var frame = chart(320, [xs[0], xs[xs.length - 1]], extent(node.concat([actual])), {
      label: "Forecast bands against what happened",
      formatY: count,
      yTitle: "incidents a day"
    });
    shade(frame, day("2020-03-01"), day("2020-06-01"), "the shift");

    frame.plot.appendChild(svg("path", {
      "class": "band", d: areaPath(frame, xs, node[0], node[levels.length - 1]), "fill-opacity": 0.15
    }));
    if (levels.length >= 4) {
      frame.plot.appendChild(svg("path", {
        "class": "band", d: areaPath(frame, xs, node[1], node[levels.length - 2]), "fill-opacity": 0.22
      }));
    }
    frame.plot.appendChild(svg("path", { "class": "median", d: linePath(frame, xs, node[middle]) }));
    frame.plot.appendChild(svg("path", { "class": "service", d: linePath(frame, xs, serviceLine) }));
    frame.plot.appendChild(svg("path", { "class": "actual-line", d: linePath(frame, xs, actual) }));

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

    legendInto(byId("fan-legend"), [
      { label: "what happened", swatch: "sw-ink" },
      { label: "the forecast (median)", swatch: "sw-accent" },
      { label: percent(levels[1]) + " to " + percent(levels[levels.length - 2]) + " and "
        + percent(levels[0]) + " to " + percent(levels[levels.length - 1]), swatch: "sw-band block" },
      { label: "staffed to " + percent(service), swatch: "sw-warn dashed" }
    ]);

    byId("fan-caption").textContent =
      payload.model + " forecasting " + payload.horizon_step + " days ahead for " +
      payload.nodes[state.node] + ", " + labels[0] + " to " + labels[labels.length - 1] +
      ". Each point is one weekly forecast read at the same distance ahead, so the line is a " +
      "record of the same task repeated, not one forecast extended.";
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

    var top = chart(260, domain, extent(covers.concat([[state.nominal]])), {
      label: "Rolling coverage against nominal",
      formatY: function (value) { return (value * 100).toFixed(0) + "%"; },
      yTitle: "outcomes inside the interval"
    });
    shade(top, day(section.shift[0]), day(section.shift[1]), "the shift");
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

    var bottom = chart(150, domain, extent(widths), {
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
      "from the same forecasts, so a method that buys coverage by widening shows it here.";
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
      "the repository, not constants in the code. An oracle that knew each day's demand would " +
      "staff " + decimals(section.oracle_units, 1) + " units a day and cost nothing.";
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
      "in the worst " + section.window_origins + " weeks, which is not a rounding error";

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
      var share = (naive.cost[at].point - best.cost[at].point) / naive.cost[at].point;
      byId("hero-saving").textContent = percent(share, 0) + " lower";
      byId("hero-saving-unit").textContent =
        "cost of being wrong, at the service level the costs imply";
    }
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
      if (payload.schema_version !== 1) {
        throw new Error("dashboard.json is schema version " + payload.schema_version +
          ", and this page reads version 1");
      }
      state.dashboard = payload;
      fillHero(payload);
      fillCoverageControls(payload.coverage);
      fillServiceControl(payload.staffing);
      drawCoverage();
      drawReconciliation();
      drawStaffing();
      return load("forecast.json");
    }).then(function (payload) {
      state.forecast = payload;
      byId("fan-step").textContent = String(payload.horizon_step);
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
