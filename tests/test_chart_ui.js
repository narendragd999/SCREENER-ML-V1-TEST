// Run with: node tests/test_chart_ui.js
// These tests execute the dashboard's own code with a minimal DOM and no network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
new vm.Script(script, {filename: 'dashboard-inline.js'});

function node(id = '') {
  const classes = new Set();
  let markup = '';
  return {id, renderHistory: [],
    get innerHTML() {return markup;},
    set innerHTML(value) {markup = value; this.renderHistory.push(value);},
    textContent: '', value: '', hidden: false, open: false,
    style: {}, dataset: {}, disabled: false,
    classList: {add: (...xs) => xs.forEach(x => classes.add(x)),
      remove: (...xs) => xs.forEach(x => classes.delete(x)), contains: x => classes.has(x)},
    setAttribute(k, v) {this[k] = v;}, getAttribute(k) {return this[k] ?? null;},
    appendChild(child) {this.innerHTML += child.innerHTML || child.textContent || '';},
    replaceChildren(...children) {this.innerHTML = children.map(c => c.innerHTML || c.textContent || '').join('');},
    scrollIntoView() {}, addEventListener() {}, querySelectorAll() {return [];}, querySelector() {return null;}};
}

function environment(fetcher, usePlotly = true) {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], node(m[1])]));
  const nav = [...html.matchAll(/<button[^>]*class="nav-btn[^"]*"[^>]*onclick="([^"]+)"[^>]*>/g)]
    .map(m => Object.assign(node(), {onclick: m[1]}));
  const timers = new Map(), timerDelays = new Map(); let nextTimer = 1;
  const plots = [];
  const context = {
    console, AbortController, URL, fetch: fetcher,
    setTimeout(fn, delay) {
      const id = nextTimer++; timers.set(id, fn); timerDelays.set(id, delay);
      // The worker's polling wait resolves immediately; request timeouts remain manual.
      if (delay === 800) queueMicrotask(() => {timerDelays.delete(id); if (timers.delete(id)) fn();});
      return id;
    },
    clearTimeout(id) {timers.delete(id); timerDelays.delete(id);},
    location: {origin: 'http://localhost:8000', href: 'http://localhost:8000/'},
    window: {addEventListener() {}, location: {origin: 'http://localhost:8000', href: 'http://localhost:8000/'}},
    document: {getElementById: id => elements.get(id) || null,
      createElement: () => node(), addEventListener() {},
      querySelector: () => node(),
      querySelectorAll: selector => selector === '.nav-btn' ? nav :
        selector === '.tab-content' ? [...elements.values()].filter(e => e.id.startsWith('tab-')) : []},
  };
  if (usePlotly) context.Plotly = {
    purge(id) {const el = typeof id === 'string' ? elements.get(id) : id; if (el) el.innerHTML = '';},
    async newPlot(id, traces, layout) {
      const el = typeof id === 'string' ? elements.get(id) : id;
      if (el) el.innerHTML = '<svg data-chart="rendered"></svg>';
      plots.push({id, traces, layout});
    }, Plots: {resize() {}},
  };
  vm.createContext(context); vm.runInContext(script, context);
  return {context, elements, plots, timers, timerDelays, nav};
}

const response = (data, ok = true, status = ok ? 200 : 503) => ({ok, status, json: async () => data});
const prices = (price = 100) => ({current_price: price, ema9: price - 1, as_of: '2026-11-09',
  fair_value: null, valuation_confidence: 'unavailable', target_3pct: price * 1.03,
  stop_loss: price - 1, target_basis: 'new_entry_estimate', candles: [
    {time: '2026-11-06', open: price - 1, high: price + 1, low: price - 2, close: price, ema9: price - 1, sma50: price - 3},
    {time: '2026-11-09', open: price, high: price + 2, low: price - 1, close: price + 1, ema9: price, sma50: price - 2}],
});
const networkFailure = async () => {throw new TypeError('Failed to fetch');};

async function main() {
  let passed = 0;
  const test = async (name, body) => {await body(); passed++; console.log('PASS', name);};
  await test('network failure identifies the local service and recovery', async () => {
    const {context} = environment(networkFailure);
    await assert.rejects(context.requestJSON('/api/chart/DIXON'), error =>
      /local|server|service/i.test(error.message) && /start|restart|run_screener|refresh/i.test(error.message));
  });
  await test('server data errors retain their diagnostic detail', async () => {
    const {context} = environment(async () => response({detail: 'DIXON price provider temporarily unavailable'}, false));
    await assert.rejects(context.requestJSON('/api/chart/DIXON'), /DIXON price provider temporarily unavailable/);
  });
  await test('structured provider errors retain a readable message', async () => {
    const {context} = environment(async () => response({error: {message: 'Daily history provider unavailable'}}));
    await assert.rejects(context.requestJSON('/api/chart/DIXON'), /Daily history provider unavailable/);
  });
  await test('a timed out request aborts and cleans up its timer', async () => {
    let signal;
    const {context, timers} = environment((url, options) => {
      signal = options.signal;
      return new Promise((resolve, reject) => signal.addEventListener('abort', () =>
        reject(Object.assign(new Error('The operation was aborted'), {name: 'AbortError'}))));
    });
    const pending = context.requestJSON('/api/chart/DIXON', {}, 2500);
    const rejected = assert.rejects(pending, error =>
      /timed out/i.test(error.message) && /retry/i.test(error.message) && /run_screener\.bat/i.test(error.message));
    assert.equal(timers.size, 1);
    [...timers.values()][0]();
    await rejected;
    assert.equal(signal.aborted, true); assert.equal(timers.size, 0);
  });
  for (const [method, ticker] of [['openStockChart', 'DIXON'], ['openSwingChartModal', 'HINDCOPPER']]) {
    await test(method + ' replaces a failed loading state with a retry', async () => {
      const {context, elements} = environment(networkFailure);
      await context[method](ticker);
      const body = elements.get('modal-plotly-chart').innerHTML;
      assert.doesNotMatch(body, /class="spinner"|Reading chart history/);
      assert.match(body, /Retry chart/i);
      assert.match(body, /local|server|service/i);
    });
  }
  await test('candlestick chart works when fair value is unavailable', async () => {
    const {context, plots, elements} = environment(async () => response(prices()));
    await context.openStockChart('DIXON');
    assert.equal(plots.length, 1); assert.equal(plots[0].traces[0].type, 'candlestick');
    assert.equal(plots[0].traces[0].close.length, 2);
    assert.match(elements.get('modal-cmp').textContent, /100/);
  });
  await test('reversal price chart works without a qualifying pattern', async () => {
    const {context, plots, elements} = environment(async () => response({...prices(430), pattern: null}));
    await context.openSwingChartModal('HINDCOPPER');
    assert.equal(plots.length, 1); assert.equal(plots[0].traces[0].type, 'candlestick');
    assert.match(elements.get('modal-cmp').textContent, /430/);
  });
  await test('a missing chart library clears loading and retains price values', async () => {
    const {context, elements} = environment(async () => response(prices()), false);
    await context.openStockChart('DIXON');
    assert.doesNotMatch(elements.get('modal-plotly-chart').innerHTML, /class="spinner"|Reading chart history/);
    assert.match(elements.get('modal-cmp').textContent, /100/);
    assert.match(elements.get('modal-plotly-chart').innerHTML, /chart library|chart engine/i);
  });
  await test('an older chart response cannot overwrite the newer symbol', async () => {
    let resolveOld;
    const {context, elements, plots} = environment(url => url.includes('OLD') ?
      new Promise(resolve => {resolveOld = resolve;}) : Promise.resolve(response(prices(200))));
    const pending = context.openStockChart('OLD');
    await context.openStockChart('NEW'); resolveOld(response(prices(999))); await pending;
    assert.equal(elements.get('modal-ticker-symbol').textContent, 'NEW');
    assert.match(elements.get('modal-cmp').textContent, /200/); assert.equal(plots.length, 1);
  });
  await test('closing the modal discards its unfinished chart response', async () => {
    let resolveChart;
    const {context, elements, plots} = environment(() => new Promise(resolve => {resolveChart = resolve;}));
    const pending = context.openStockChart('DIXON'); context.closeModal();
    resolveChart(response(prices())); await pending;
    assert.equal(elements.get('chart-modal').classList.contains('active'), false);
    assert.equal(plots.length, 0);
  });
  await test('a valuation request failure clears every stale chart and result', async () => {
    const {context, elements, timerDelays} = environment(networkFailure);
    elements.get('fv-input-ticker').value = 'DIXON';
    const chartIds = ['fv-plotly-chart', 'fv-chart-op', 'fv-chart-sales', 'fv-chart-ttm', 'fv-chart-ttm-sales'];
    const resultIds = ['fv-models-grid', 'fv-annual-tbody', 'fv-quarterly-tbody',
      'fv-details-op', 'fv-details-sales', 'fv-details-ttm', 'fv-op-badge-container', 'fv-sales-badge-container',
      'fv-sel-fy-op', 'fv-sel-fy-sales', 'fv-sel-q-ttm'];
    for (const id of chartIds) elements.get(id).innerHTML = '<svg data-old-symbol="OLD"></svg>';
    for (const id of ['fv-cmp', 'fv-composite-price', 'fv-gap-text']) elements.get(id).textContent = 'OLD result';
    for (const id of resultIds) elements.get(id).innerHTML = 'OLD result';
    elements.get('fv-bank-badge').style.display = 'inline-flex';
    vm.runInContext('currentFVData = {ticker: "OLD"}', context);
    await context.lookupFairValue();
    assert.equal(vm.runInContext('currentFVData', context), null);
    for (const id of chartIds) {
      assert.doesNotMatch(elements.get(id).innerHTML, /<svg|class="spinner"|Reading valuation evidence|OLD/);
      assert.match(elements.get(id).innerHTML, /Retry valuation/);
      assert.match(elements.get(id).innerHTML, /Open price chart/);
    }
    for (const id of ['fv-cmp', 'fv-composite-price', 'fv-gap-text']) assert.equal(elements.get(id).textContent, '—');
    for (const id of resultIds) assert.equal(elements.get(id).innerHTML, '');
    assert.equal(elements.get('fv-bank-badge').style.display, 'none');
    assert.equal(elements.get('fv-status-badge').textContent, 'UNAVAILABLE');
    assert.match(elements.get('fv-quality-note').innerHTML, /local service/i);
    assert.ok(![...timerDelays.values()].includes(180000), 'Valuation request timeout was not cleared');
  });
  await test('absent valuation gaps remain unavailable across headlines, tables, and snapshots', async () => {
    const data = {ticker: 'DIXON', status: 'PARTIAL', current_price: 120, composite_fair_price: null,
      composite_gain_pct: null, valuation_bucket: 'FAIR', models: {},
      fy_rows: [{fy: 2025, comp_gain: null, misprice: null, bucket: 'FAIR'}],
      quarterly_rows: [{quarter: 'Sep 2025', misprice_pct: null, status: 'FAIR'}]};
    const {context, elements} = environment(async () => response(data));
    elements.get('fv-input-ticker').value = 'DIXON';
    await context.lookupFairValue();
    assert.equal(vm.runInContext('currentFVData', context)?.ticker, 'DIXON');
    assert.equal(elements.get('fv-status-badge').textContent, 'UNAVAILABLE');
    assert.match(elements.get('fv-gap-text').textContent, /No supported valuation gap/);
    assert.equal(elements.get('fv-cmp').textContent, '₹120.00');
    for (const id of ['fv-annual-tbody', 'fv-quarterly-tbody']) {
      assert.match(elements.get(id).innerHTML, /UNAVAILABLE/);
      assert.doesNotMatch(elements.get(id).innerHTML, /0\.0%|>FAIR</);
    }
    for (const key of ['op', 'sales', 'ttm']) {
      context.inspectFVSnapshot(key, 'missing-period');
      assert.match(elements.get('fv-details-' + key).innerHTML, /UNAVAILABLE/);
      assert.doesNotMatch(elements.get('fv-details-' + key).innerHTML, /0\.0%|>FAIR</);
    }
    for (const id of ['fv-chart-op', 'fv-chart-sales', 'fv-chart-ttm', 'fv-chart-ttm-sales']) {
      assert.match(elements.get(id).innerHTML, /No .* chart history/);
      assert.doesNotMatch(elements.get(id).innerHTML, /class="spinner"/);
    }
  });
  await test('a measured zero valuation gap remains a valid zero', async () => {
    const {context, elements} = environment(async () => response({}));
    context.renderFVAnnualTable([{fy: 2025, comp_gain: 0, bucket: 'FAIR'}]);
    context.renderFVQuarterlyTable([{quarter: 'Sep 2025', misprice_pct: 0, status: 'FAIR'}]);
    vm.runInContext('currentFVData = {models: {op: {historical: {"2025": {misprice_pct: 0}}}}}', context);
    context.inspectFVSnapshot('op', '2025');
    for (const id of ['fv-annual-tbody', 'fv-quarterly-tbody', 'fv-details-op']) {
      assert.match(elements.get(id).innerHTML, /0\.0%/);
      assert.match(elements.get(id).innerHTML, />FAIR</);
      assert.doesNotMatch(elements.get(id).innerHTML, /UNAVAILABLE/);
    }
  });
  for (const failure of ['missing library', 'synchronous rendering error', 'asynchronous rendering error']) {
    await test('valuation preserves numbers and tables after ' + failure, async () => {
      const {context, elements} = environment(async () => response({}), failure !== 'missing library');
      if (failure === 'synchronous rendering error') context.Plotly.newPlot = () => {throw new Error('Renderer failed');};
      if (failure === 'asynchronous rendering error') context.Plotly.newPlot = async () => {throw new Error('Renderer failed');};
      elements.get('fv-cmp').textContent = '₹120.00';
      elements.get('fv-annual-tbody').innerHTML = '<tr><td>Annual fundamentals</td></tr>';
      elements.get('fv-plotly-chart').innerHTML = '<div class="spinner"></div>';
      await context.drawValuationChart('fv-plotly-chart', [], {}, {});
      assert.doesNotMatch(elements.get('fv-plotly-chart').innerHTML, /class="spinner"/);
      assert.match(elements.get('fv-plotly-chart').innerHTML, /Chart rendering unavailable/);
      assert.match(elements.get('fv-plotly-chart').innerHTML, /numbers and tables remain available/);
      assert.equal(elements.get('fv-cmp').textContent, '₹120.00');
      assert.match(elements.get('fv-annual-tbody').innerHTML, /Annual fundamentals/);
    });
  }
  await test('valuation year windows exclude older calendar dates and retain initialized EMA', async () => {
    const {context, plots} = environment(async () => response({}));
    const history=[{date:'2024-11-09',close:10},{date:'2025-11-08',close:20},
      {date:'2025-11-10',close:30},{date:'2026-06-01',close:40},{date:'2026-11-09',close:50}];
    vm.runInContext('currentFVWindowDays=365',context);
    context.renderFVPlotlyChart({price_history:history,fy_rows:[]});
    const chart=plots.find(p=>p.id==='fv-plotly-chart');
    assert.deepEqual(Array.from(chart.traces[0].x),['2025-11-10','2026-06-01','2026-11-09']);
    assert.deepEqual(Array.from(chart.traces[1].y),Array.from(context.calc9EMA(history.map(p=>p.close))).slice(-3));
  });
  await test('scan progress shows actual worker counts with an indeterminate queued state', async () => {
    const {context, elements} = environment(async () => response({}));
    context.displayJobProgress('engine-status', {status: 'queued', progress: {phase: 'fetching_prices', percent: null}});
    const queued = elements.get('engine-status').innerHTML;
    assert.match(queued, /fetching prices/); assert.match(queued, /Preparing work/);
    assert.doesNotMatch(queued, /<progress[^>]* value=/);
    context.displayJobProgress('engine-status', {status: 'running', progress: {
      phase: 'fetching_prices', completed: 2, total: 6, percent: 33.3,
      ticker: '<DIXON>', failed_count: 1, message: 'Reading completed daily bars',
    }});
    const running = elements.get('engine-status').innerHTML;
    assert.match(running, /2 \/ 6 · 33%/);
    assert.match(running, /<progress max="100" value="33\.3"/);
    assert.match(running, /Current symbol: &lt;DIXON&gt;/);
    assert.match(running, /Unavailable: 1/);
    assert.match(running, /Reading completed daily bars/);
    assert.equal(elements.get('engine-status').hidden, false);
  });
  await test('a progress job returns its result and restores original control states', async () => {
    const calls = [];
    const jobs = [
      {job_id: 'job/7', status: 'queued', progress: {phase: 'Preparing job', percent: null}},
      {job_id: 'job/7', status: 'running', progress: {phase: 'Analyzing prices', completed: 2, total: 6, percent: 33.3}},
      {job_id: 'job/7', status: 'completed', progress: {phase: 'Completed', completed: 6, total: 6, percent: 100}, result: {stocks: ['DIXON']}},
    ];
    const {context, elements, timers} = environment(async (url, options) => {
      calls.push({url, options}); return response(jobs.shift());
    });
    const enabled = node(), alreadyDisabled = Object.assign(node(), {disabled: true});
    elements.get('tab-screener').querySelectorAll = () => [enabled, alreadyDisabled];
    const pending = context.runProgressJob('screener', {tickers: ['DIXON']}, 'engine-status');
    assert.equal(enabled.disabled, true); assert.equal(alreadyDisabled.disabled, true);
    assert.deepEqual(await pending, {stocks: ['DIXON']});
    assert.equal(enabled.disabled, false); assert.equal(alreadyDisabled.disabled, true);
    assert.equal(calls[0].url, '/api/jobs/screener');
    assert.equal(calls[0].options.method, 'POST');
    assert.deepEqual(JSON.parse(calls[0].options.body), {tickers: ['DIXON']});
    assert.ok(calls.slice(1).every(call => call.url === '/api/jobs/job%2F7'));
    const history = elements.get('engine-status').renderHistory;
    assert.ok(history.some(markup => /2 \/ 6 · 33%/.test(markup)));
    assert.match(history.at(-1), /6 \/ 6 · 100%/);
    assert.equal(timers.size, 0);
  });
  await test('a failed progress job reports the provider reason and restores controls', async () => {
    const jobs = [
      {job_id: 'failed-job', status: 'queued', progress: {percent: null}},
      {job_id: 'failed-job', status: 'failed', progress: {completed: 1, total: 6, percent: 16.7}, error: {message: 'Price provider unavailable for DIXON'}},
      {job_id: 'retry-job', status: 'completed', progress: {percent: 100}, result: {stocks: ['DIXON']}},
    ];
    const {context, elements, timers} = environment(async () => response(jobs.shift()));
    const control = node(); elements.get('tab-screener').querySelectorAll = () => [control];
    await assert.rejects(context.runProgressJob('screener', {}, 'engine-status'), /Price provider unavailable for DIXON/);
    assert.equal(control.disabled, false); assert.equal(timers.size, 0);
    assert.equal(vm.runInContext('activeProgressJobs.size', context), 0);
    assert.match(elements.get('engine-status').innerHTML, /Job stopped/);
    assert.match(elements.get('engine-status').innerHTML, /Price provider unavailable for DIXON/);
    assert.deepEqual(await context.runProgressJob('screener', {}, 'engine-status'), {stocks: ['DIXON']});
    assert.equal(control.disabled, false);
  });
  await test('duplicate scan clicks share one progress job and one promise', async () => {
    let submit, submissions = 0;
    const {context, elements} = environment(() => {
      submissions++;
      return new Promise(resolve => {submit = resolve;});
    });
    const control = node(); elements.get('tab-screener').querySelectorAll = () => [control];
    const first = context.runProgressJob('screener', {tickers: ['DIXON']}, 'engine-status');
    const duplicate = context.runProgressJob('screener', {tickers: ['DIXON']}, 'engine-status');
    assert.equal(first, duplicate);
    assert.equal(submissions, 1);
    submit(response({job_id: 'single-job', status: 'completed', progress: {percent: 100}, result: {stocks: ['DIXON']}}));
    assert.deepEqual(await first, {stocks: ['DIXON']}); await duplicate;
    assert.equal(control.disabled, false);
    assert.equal(vm.runInContext('activeProgressJobs.size', context), 0);
  });
  await test('duration strategies use different holding and trend defaults', async () => {
    const {context, elements} = environment(async () => response({}));
    elements.get('bt-strategy').value='duration_next_open';
    context.changeBacktestStrategy();
    assert.equal(elements.get('bt-maxhold').value,'90');
    assert.equal(elements.get('bt-maxhold').max,'250');
    assert.equal(elements.get('bt-uptrend').checked,false);
    assert.match(elements.get('bt-mode-note').textContent,/No stop/);
    context.loadDurationCandidate();
    assert.equal(elements.get('bt-strategy').value,'duration_next_open');
    assert.equal(elements.get('bt-avoid-resistance').checked,true);
    assert.equal(elements.get('bt-pullback').checked,false);
    elements.get('bt-strategy').value='confirmed_ema';
    context.changeBacktestStrategy();
    assert.equal(elements.get('bt-maxhold').value,'20');
    assert.equal(elements.get('bt-maxhold').max,'60');
    assert.equal(elements.get('bt-uptrend').checked,true);
    assert.equal(elements.get('bt-snapshot-label').style.display,'none');
  });
  await test('duration output preserves all denominators and unresolved pending exits', async () => {
    const {context, elements, plots} = environment(async () => response({}));
    const trades=[
      {ticker:'TEST',entry_date:'2026-01-01',entry_price:100,target_price:103,status:'TARGET_HIT',net_pnl_pct:2.7,full_horizon_available:true},
      {ticker:'TEST',entry_date:'2026-01-02',entry_price:100,target_price:103,status:'TIME_EXIT',net_pnl_pct:-8,full_horizon_available:true},
      {ticker:'OTHER',entry_date:'2026-11-09',entry_price:100,target_price:103,status:'TIME_EXIT_PENDING',net_pnl_pct:null,full_horizon_available:true}
    ];
    await context.renderDurationBacktest({parameters:{mode:'duration_next_open',max_hold_sessions:90,target_pct:3},
      stats:{total_entries:3,closed_count:2,open_count:1,censored_count:0,mature_entries:3,mature_target_hits:1,target_hits:1,
        mature_target_hit_rate_pct:33.33,all_entry_target_hit_rate_pct:33.33,net_win_rate_pct:50,
        horizon_target_curve:[{sessions:90,target_hit_rate_pct:33.33}]},trades,
      retrospective_current_fair_value_cohort:{available:false,label:'Unrequested FV cohort'}});
    assert.equal(elements.get('bt-duration-results').style.display,'block');
    assert.match(elements.get('bt-duration-results').innerHTML,/2 closed \/ 1 unresolved/);
    assert.doesNotMatch(elements.get('bt-duration-results').innerHTML,/Unrequested FV cohort/);
    assert.equal(plots.length,1);
    assert.equal(context.durationMatchingTrades().length,3);
    elements.get('bt-duration-outcome').value='nonwinner';
    assert.equal(context.durationMatchingTrades().length,1);
    assert.equal(context.durationMatchingTrades()[0].status,'TIME_EXIT');
    elements.get('bt-duration-outcome').value='open';
    assert.equal(context.durationMatchingTrades()[0].status,'TIME_EXIT_PENDING');
    elements.get('bt-duration-search').value='TEST';
    assert.equal(context.durationMatchingTrades().length,0);
  });
  await test('a failed duration chart preserves measured results and exports', async () => {
    const {context,elements}=environment(async()=>response({}));
    context.Plotly.newPlot=async()=>{throw new Error('Renderer failed');};
    await context.renderDurationBacktest({parameters:{max_hold_sessions:90,target_pct:3},
      stats:{mature_entries:10,mature_target_hits:9,mature_target_hit_rate_pct:90,
        horizon_target_curve:[{sessions:90,target_hit_rate_pct:90}]},trades:[]});
    assert.match(elements.get('bt-duration-chart').innerHTML,/Renderer failed/);
    assert.match(elements.get('bt-duration-results').innerHTML,/90.0%/);
    assert.match(elements.get('bt-duration-results').innerHTML,/Export full CSV/);
  });
  await test('Help is reachable and explains the complete research workflow', async () => {
    const {context, elements, nav} = environment(async () => response({}));
    context.switchTab('help');
    assert.equal(elements.get('tab-help')?.classList.contains('active'), true);
    assert.ok(nav.some(button => button.onclick.includes('help') && button.classList.contains('active')));
    const help = html.match(/<section id="tab-help"[\s\S]*?<\/section>/)?.[0] || '';
    for (const phrase of ['Opportunities', 'EMA screener', 'Reversals', 'Valuation', 'Paper journal',
      'Backtests', 'risk', 'ATR', 'RSI', 'volume', 'relative strength', 'TARGET', 'STOP', 'first 60'])
      assert.ok(help.toLowerCase().includes(phrase.toLowerCase()), 'Help missing ' + phrase);
  });
  console.log(`${passed} chart and help UI regression tests passed.`);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
