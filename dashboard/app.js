/* The dashboard, drawn from two JSON files and nothing else.
 *
 * No framework and no charting library: the page draws four panels, and a library big
 * enough to draw them is bigger than the measurements it would draw. Everything here is
 * SVG built with the DOM API, which also means the content security policy can forbid
 * inline scripts and styles outright rather than carrying an exception for a bundle.
 *
 * The one rule worth knowing before changing anything: this file computes nothing that the
 * backtest could have computed. The rolling mean below is the exception, and it mirrors
 * headroom.report.charts.rolling_mean exactly, so that the chart in the README and the
 * chart on this page agree at the default window.
 */
"use strict";

(function () {
  var NS = "http://www.w3.org/2000/svg";
  var DAY = 86400000;

  /* The methods' colours, from src/headroom/report/charts.py. */
  var LINES = ["#1f77b4", "#d95f02", "#7570b3", "#117733"];

  var state = {
    dashboard: null,
    forecast: null,
    node: 0,
    range: "shift",
    nominal: null,
    window: 13,
    service: 0
  };

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

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  function day(iso) { return Date.parse(iso + "T00:00:00Z") / DAY; }

  function isNumber(value) { return typeof value === "number" && isFinite(value); }

  /* Formatting ------------------------------------------------------------------ */

  function decimals(value, places) {
    return isNumber(value) ? value.toFixed(places) : "n/a";
  }

  function signed(value, places) {
    if (!isNumber(value)) { return "n/a"; }
    return (value > 0 ? "+" : "") + value.toFixed(places);
  }

  function percent(value, places) {
    return isNumber(value) ? (value * 100).toFixed(places || 0) + "%" : "n/a";
  }

  /* An estimate is never shown without its interval: the interval is the measurement. */
  function estimateCell(estimate, places, showSign) {
    var cell = el("td");
    var format = showSign ? signed : decimals;
    cell.appendChild(el("span", null, format(estimate.point, places)));
    cell.appendChild(document.createTextNode(" "));
    cell.appendChild(el(
      "span",
      { "class": "interval" },
      "[" + format(estimate.low, places) + ", " + format(estimate.high, places) + "]"
    ));
    if (showSign && isNumber(estimate.point)) {
      cell.firstChild.className = estimate.point < 0 ? "better" : "worse";
    }
    return cell;
  }

  /* Charts ---------------------------------------------------------------------- */

  function ticks(lo, hi, count) {
    var span = hi - lo;
    if (!(span > 0)) { return [lo]; }
    var raw = span / count;
    var magnitude = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = magnitude;
    var steps = [1, 2, 2.5, 5, 10];
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] * magnitude >= raw) { step = steps[i] * magnitude; break; }
    }
    var out = [];
    for (var t = Math.ceil(lo / step) * step; t <= hi + step / 1e6; t += step) { out.push(t); }
    return out;
  }

  function yearTicks(lo, hi) {
    var first = new Date(lo * DAY).getUTCFullYear();
    var last = new Date(hi * DAY).getUTCFullYear();
    var every = Math.max(1, Math.ceil((last - first + 1) / 8));
    var out = [];
    for (var year = first; year <= last; year += every) {
      var at = Date.UTC(year, 0, 1) / DAY;
      if (at >= lo && at <= hi) { out.push({ at: at, label: String(year) }); }
    }
    if (out.length < 2) {
      out = [
        { at: lo, label: new Date(lo * DAY).toISOString().slice(0, 7) },
        { at: hi, label: new Date(hi * DAY).toISOString().slice(0, 7) }
      ];
    }
    return out;
  }

  function chart(height, xDomain, yDomain, options) {
    var width = 880;
    var margin = { top: 12, right: 16, bottom: 26, left: 52 };
    var inner = width - margin.left - margin.right;
    var deep = height - margin.top - margin.bottom;
    var root = svg("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": (options && options.label) || "chart"
    });
    var plot = svg("g", { transform: "translate(" + margin.left + "," + margin.top + ")" });
    root.appendChild(plot);

    var xSpan = xDomain[1] - xDomain[0] || 1;
    var ySpan = yDomain[1] - yDomain[0] || 1;
    var frame = {
      root: root,
      plot: plot,
      inner: inner,
      deep: deep,
      x: function (value) { return (value - xDomain[0]) / xSpan * inner; },
      y: function (value) { return deep - (value - yDomain[0]) / ySpan * deep; }
    };

    var axes = svg("g", { "class": "axis" });
    plot.appendChild(axes);
    ticks(yDomain[0], yDomain[1], 4).forEach(function (value) {
      var at = frame.y(value);
      axes.appendChild(svg("line", {
        "class": "gridline", x1: 0, x2: inner, y1: at, y2: at
      }));
      var text = svg("text", { x: -8, y: at + 4, "text-anchor": "end" });
      text.textContent = (options && options.formatY) ? options.formatY(value) : String(value);
      axes.appendChild(text);
    });
    yearTicks(xDomain[0], xDomain[1]).forEach(function (tick) {
      var at = frame.x(tick.at);
      axes.appendChild(svg("line", { x1: at, x2: at, y1: deep, y2: deep + 4 }));
      var text = svg("text", { x: at, y: deep + 17, "text-anchor": "middle" });
      text.textContent = tick.label;
      axes.appendChild(text);
    });
    axes.appendChild(svg("line", { x1: 0, x2: inner, y1: deep, y2: deep }));
    return frame;
  }

  function shade(frame, from, to, xDomain) {
    var lo = Math.max(from, xDomain[0]);
    var hi = Math.min(to, xDomain[1]);
    if (hi <= lo) { return; }
    frame.plot.appendChild(svg("rect", {
      "class": "shift-span",
      x: frame.x(lo), y: 0, width: frame.x(hi) - frame.x(lo), height: frame.deep
    }));
  }

  /* Both of these break the path wherever a value is missing, rather than drawing a
     straight line across a gap as though something had been measured there. */
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
      if (run.length < 2) { run = []; return; }
      var ahead = [];
      var back = [];
      run.forEach(function (i) {
        ahead.push(frame.x(xs[i]).toFixed(1) + " " + frame.y(lower[i]).toFixed(1));
      });
      for (var j = run.length - 1; j >= 0; j--) {
        var i = run[j];
        back.push(frame.x(xs[i]).toFixed(1) + " " + frame.y(upper[i]).toFixed(1));
      }
      parts.push("M" + ahead.join("L") + "L" + back.join("L") + "Z");
      run = [];
    }
    for (var i = 0; i < xs.length; i++) {
      if (isNumber(lower[i]) && isNumber(upper[i])) { run.push(i); } else { flush(); }
    }
    flush();
    return parts.join(" ");
  }

  function legend(frame, entries) {
    var group = svg("g", { "class": "legend", transform: "translate(4,2)" });
    var offset = 0;
    entries.forEach(function (entry) {
      var line = svg("line", {
        x1: offset, x2: offset + 16, y1: 8, y2: 8,
        stroke: entry.colour, "stroke-width": 2
      });
      if (entry.dash) { line.setAttribute("stroke-dasharray", entry.dash); }
      group.appendChild(line);
      var text = svg("text", { x: offset + 21, y: 12 });
      text.textContent = entry.label;
      group.appendChild(text);
      offset += 30 + entry.label.length * 6.2;
    });
    frame.plot.appendChild(group);
  }

  /* The rolling mean, as headroom.report.charts.rolling_mean computes it: trailing, and
     only where the whole window was measured. A window that is part gap is not an average
     of anything. */
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

  function extent(serieses) {
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
    var pad = (hi - lo) * 0.06 || 1;
    return [lo - pad, hi + pad];
  }

  /* The forecast panel ---------------------------------------------------------- */

  function serviceQuantile(bands, levels, wanted, at) {
    var lo = 0;
    while (lo < levels.length - 2 && levels[lo + 1] < wanted) { lo++; }
    var hi = lo + 1;
    var low = bands[lo][at];
    var high = bands[hi][at];
    if (!isNumber(low) || !isNumber(high)) { return null; }
    var span = levels[hi] - levels[lo];
    var share = span > 0 ? (Math.min(Math.max(wanted, levels[0]), levels[levels.length - 1]) - levels[lo]) / span : 0;
    return low + (high - low) * share;
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
    var host = document.getElementById("fan-chart");
    clear(host);
    if (!payload) { return; }

    var span = fanWindow(payload.days);
    var xs = [];
    for (var i = span[0]; i <= span[1]; i++) { xs.push(day(payload.days[i])); }
    var node = payload.bands[state.node].map(function (level) {
      return level.slice(span[0], span[1] + 1);
    });
    var actual = payload.actual[state.node].slice(span[0], span[1] + 1);
    var levels = payload.levels;
    var middle = Math.floor(levels.length / 2);

    var service = state.dashboard
      ? state.dashboard.staffing.service_levels[state.service]
      : null;
    var serviceLine = [];
    if (service !== null) {
      for (var k = 0; k < xs.length; k++) {
        serviceLine.push(serviceQuantile(node, levels, service, k));
      }
    }

    var frame = chart(300, [xs[0], xs[xs.length - 1]], extent(node.concat([actual])), {
      label: "Forecast bands against what happened",
      formatY: function (value) { return Math.round(value).toLocaleString(); }
    });
    shade(frame, day("2020-03-01"), day("2020-06-01"), [xs[0], xs[xs.length - 1]]);

    var outer = svg("path", {
      "class": "band", d: areaPath(frame, xs, node[0], node[levels.length - 1]),
      "fill-opacity": 0.18
    });
    frame.plot.appendChild(outer);
    if (levels.length >= 4) {
      frame.plot.appendChild(svg("path", {
        "class": "band", d: areaPath(frame, xs, node[1], node[levels.length - 2]),
        "fill-opacity": 0.28
      }));
    }
    frame.plot.appendChild(svg("path", { "class": "median", d: linePath(frame, xs, node[middle]) }));
    if (serviceLine.length) {
      frame.plot.appendChild(svg("path", { "class": "service", d: linePath(frame, xs, serviceLine) }));
    }
    frame.plot.appendChild(svg("path", { "class": "actual-line", d: linePath(frame, xs, actual) }));
    host.appendChild(frame.root);

    var name = payload.nodes[state.node];
    var caption = payload.model + ", " + payload.horizon_step + " days ahead, for " + name +
      ". Shaded: the " + percent(levels[0]) + " to " + percent(levels[levels.length - 1]) +
      " band, with " + percent(levels[1]) + " to " + percent(levels[levels.length - 2]) +
      " inside it. Blue line: the median. Black line: what happened. Orange dashes: the " +
      percent(service) + " quantile the rota would be staffed to, interpolated between the " +
      "bands above for display. The shaded column is March to June 2020.";
    document.getElementById("fan-caption").textContent = caption;
  }

  /* The coverage panel ---------------------------------------------------------- */

  function drawCoverage() {
    var section = state.dashboard.coverage;
    var host = document.getElementById("coverage-chart");
    clear(host);
    var xs = section.days.map(day);
    var domain = [xs[0], xs[xs.length - 1]];
    var chosen = section.series.filter(function (one) {
      return Math.abs(one.nominal - state.nominal) < 1e-9;
    });
    var covers = chosen.map(function (one) { return rolling(one.coverage, state.window); });
    var widths = chosen.map(function (one) { return rolling(one.width, state.window); });

    var top = chart(250, domain, extent(covers.concat([[state.nominal]])), {
      label: "Rolling coverage against nominal",
      formatY: function (value) { return (value * 100).toFixed(0) + "%"; }
    });
    shade(top, day(section.shift[0]), day(section.shift[1]), domain);
    top.plot.appendChild(svg("line", {
      "class": "nominal", x1: 0, x2: top.inner, y1: top.y(state.nominal), y2: top.y(state.nominal)
    }));
    covers.forEach(function (values, i) {
      top.plot.appendChild(svg("path", {
        d: linePath(top, xs, values), fill: "none", stroke: LINES[i % LINES.length],
        "stroke-width": 1.4
      }));
    });
    legend(top, chosen.map(function (one, i) {
      return { label: one.method, colour: LINES[i % LINES.length] };
    }).concat([{ label: "nominal", colour: "currentColor", dash: "5 4" }]));
    host.appendChild(top.root);

    var bottom = chart(150, domain, extent(widths), {
      label: "Mean interval width",
      formatY: function (value) { return Math.round(value).toLocaleString(); }
    });
    shade(bottom, day(section.shift[0]), day(section.shift[1]), domain);
    widths.forEach(function (values, i) {
      bottom.plot.appendChild(svg("path", {
        d: linePath(bottom, xs, values), fill: "none", stroke: LINES[i % LINES.length],
        "stroke-width": 1.4
      }));
    });
    host.appendChild(bottom.root);

    document.getElementById("coverage-caption").textContent =
      "Above: the share of outcomes inside the " + percent(state.nominal) +
      " interval, in a trailing window of " + state.window + " weekly origins, at the " +
      section.level + " level. The dashed line is what was promised. Below: the mean width of " +
      "those same intervals, in incidents a day. Shaded: March to June 2020.";
  }

  /* The reconciliation panel ---------------------------------------------------- */

  function drawReconciliation() {
    var section = state.dashboard.reconciliation;
    var table = document.getElementById("reconciliation-table");
    var caption = table.querySelector("caption");
    clear(table);
    table.appendChild(caption);

    var variants = section.rows[0].variants.map(function (one) { return one.name; });
    var head = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", { scope: "col" }, "Level"));
    headRow.appendChild(el("th", { scope: "col" }, "CRPS, " + section.model));
    variants.forEach(function (name) {
      headRow.appendChild(el("th", { scope: "col" }, name + ": change in CRPS"));
      headRow.appendChild(el(
        "th", { scope: "col" }, name + ": coverage at " + percent(section.nominal)
      ));
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

    document.getElementById("reconciliation-lede").textContent =
      "Forecasts made one series at a time do not add up: the boroughs here miss their city " +
      "by up to " + decimals(section.base_coherence_error, 1) + " incidents. MinT projects " +
      "them onto the forecasts that do add up, and this is what that cost or bought. The " +
      "largest breach left afterwards is " + section.reconciled_coherence_error.toExponential(1) +
      ", which is rounding.";

    document.getElementById("reconciliation-note").textContent =
      "MinT reconciles the median and carries each node's spread with it. MinT paths " +
      "reconciles the whole distribution: every draw is put through the same projection, so " +
      "every draw adds up across all series at once, which is what a decision taken over the " +
      "whole hierarchy needs. Its quantiles are not supposed to sum, because the boroughs do " +
      "not have their bad days together. Paths are not floored at zero, because that would " +
      "break the coherence they exist for: " + percent(section.negative_share, 3) +
      " of path values fall below it.";
  }

  /* The staffing panel ---------------------------------------------------------- */

  function drawStaffing() {
    var section = state.dashboard.staffing;
    var level = section.service_levels[state.service];
    var table = document.getElementById("staffing-table");
    var caption = table.querySelector("caption");
    clear(table);
    table.appendChild(caption);

    var head = el("thead");
    var headRow = el("tr");
    ["Method", "Units staffed a day", "Cost a day against the oracle",
      "Cost against " + section.reference].forEach(function (name) {
      headRow.appendChild(el("th", { scope: "col" }, name));
    });
    head.appendChild(headRow);
    table.appendChild(head);

    var body = el("tbody");
    section.methods.forEach(function (method) {
      var line = el("tr");
      if (method.name === section.reference) { line.className = "reference"; }
      line.appendChild(el("th", { scope: "row" }, method.name));
      line.appendChild(estimateCell(method.units[state.service], 1, false));
      line.appendChild(estimateCell(method.cost[state.service], 2, false));
      var change = method.cost_change[state.service];
      line.appendChild(change ? estimateCell(change, 2, true) : el("td", null, "reference"));
      body.appendChild(line);
    });
    table.appendChild(body);

    var implied = section.inputs.implied_service_level;
    document.getElementById("service-level-out").textContent =
      percent(level) + (Math.abs(level - implied) < 1e-9 ? " (implied by the costs)" : "");
    document.getElementById("staffing-lede").textContent =
      "One unit serves " + section.inputs.demand_per_unit + " incidents a day. A spare " +
      "unit-day costs " + section.inputs.cost_over + " and a missing one " +
      section.inputs.cost_under + ", so the costs imply staffing to the " + percent(implied) +
      " quantile of the forecast. Those three numbers are a table in the repository and not " +
      "constants in the code; every number below moves when they are replaced. An oracle that " +
      "knew each day's demand would staff " + decimals(section.oracle_units, 1) +
      " units a day and cost nothing.";
  }

  /* Controls -------------------------------------------------------------------- */

  function fillNodes(payload) {
    var select = document.getElementById("fan-node");
    var groups = {};
    var titles = { city: "City", borough: "Boroughs", area: "Dispatch areas" };
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
    document.getElementById("fan-range").addEventListener("change", function (event) {
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
    var select = document.getElementById("coverage-nominal");
    nominals.forEach(function (nominal) {
      var option = el("option", { value: String(nominal) }, percent(nominal));
      if (nominal === state.nominal) { option.setAttribute("selected", "selected"); }
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      state.nominal = Number(select.value);
      drawCoverage();
    });

    var slider = document.getElementById("coverage-window");
    state.window = section.window_origins;
    slider.value = String(state.window);
    document.getElementById("coverage-window-out").textContent = state.window + " origins";
    slider.addEventListener("input", function () {
      state.window = Number(slider.value);
      document.getElementById("coverage-window-out").textContent = state.window + " origins";
      drawCoverage();
    });
  }

  function fillServiceControl(section) {
    var slider = document.getElementById("service-level");
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

  /* Start ----------------------------------------------------------------------- */

  function fail(message) {
    var box = document.getElementById("failure");
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
      var origins = payload.origins;
      var trained = origins.train_window_days
        ? "from a training window of " + origins.train_window_days + " days"
        : "from every day before it";
      document.getElementById("provenance").textContent =
        origins.count.toLocaleString() + " weekly forecast origins, " + origins.first + " to " +
        origins.last + ", each forecasting the next " + origins.horizon_days + " days for " +
        payload.hierarchy.nodes.length + " series, " + trained + ". Every interval on this " +
        "page is a 95 percent moving-block bootstrap interval over those origins. Exported " +
        payload.generated + ".";
      fillCoverageControls(payload.coverage);
      fillServiceControl(payload.staffing);
      drawCoverage();
      drawReconciliation();
      drawStaffing();
      return load("forecast.json");
    }).then(function (payload) {
      state.forecast = payload;
      document.getElementById("fan-step").textContent = String(payload.horizon_step);
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
