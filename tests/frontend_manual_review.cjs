// Optional focused browser QA. All input media must be synthetic.
// CASHWATCH_PLAYWRIGHT_MODULE=/path/to/playwright CASHWATCH_SYNTHETIC_MOV=/path/to/fixture.mov
// CASHWATCH_SYNTHETIC_MP4=/path/to/fixture-vp9.mp4 node tests/frontend_manual_review.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const {createHash} = require('node:crypto');
const {chromium} = require(process.env.CASHWATCH_PLAYWRIGHT_MODULE || 'playwright');
const web = path.resolve(__dirname, '../src/cashwatch/web');
const output = process.env.CASHWATCH_UI_QA_OUT || fs.mkdtempSync(path.join(os.tmpdir(), 'cashwatch-manual-qa-'));
const mov = fs.readFileSync(process.env.CASHWATCH_SYNTHETIC_MOV);
const mp4 = fs.readFileSync(process.env.CASHWATCH_SYNTHETIC_MP4);
const second = Buffer.concat([mp4, Buffer.from('\nsynthetic-distinct-fixture')]);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const results = [];
const errors = [];
const requests = [];
const testKey = 'synthetic-manual-review-qa-key';
const fileCapabilities = {max_upload_mb: 100, accepted_extensions: ['.mov', '.mp4']};
let capabilities = {...fileCapabilities, model_configured: false, active_rules: 0};
let origin;
function passed(name) { results.push({name, passed: true}); process.stdout.write(`PASS ${name}\n`); }
async function until(predicate, name) {
  const end = Date.now() + 10000;
  while (Date.now() < end) { if (await predicate()) return; await new Promise(resolve => setTimeout(resolve, 40)); }
  throw new Error(name);
}
async function choose(page, name, buffer, mimeType = 'video/mp4') {
  await page.locator('#video-input').setInputFiles({name, mimeType, buffer});
}
async function downloaded(page) {
  const promise = page.waitForEvent('download');
  await page.locator('#manual-export').click();
  const download = await promise;
  assert.match(download.suggestedFilename(), /^cashwatch_manual_review_v1-[a-f0-9]{12}\.json$/);
  return JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
}
async function watch(context) {
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', req => requests.push({url: req.url(), method: req.method()}));
  await page.goto(origin, {waitUntil: 'networkidle'});
  return page;
}
const server = http.createServer((req, res) => {
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
  if (req.method !== 'GET') { res.writeHead(500); return res.end('Unexpected upload'); }
  if (req.url === '/healthz') { res.setHeader('Content-Type', 'application/json'); return res.end(JSON.stringify({status: 'alive'})); }
  if (req.url === '/api/v1/capabilities') {
    res.setHeader('Content-Type', 'application/json');
    if (req.headers['x-api-key'] !== testKey) {
      res.statusCode = 401;
      return res.end(JSON.stringify({detail: 'Неверный или отсутствующий X-API-Key'}));
    }
    return res.end(JSON.stringify(capabilities));
  }
  const names = {'/': 'index.html', '/static/app.js': 'app.js', '/static/styles.css': 'styles.css'};
  const name = names[req.url];
  if (!name) { res.writeHead(404); return res.end(); }
  res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
  res.end(fs.readFileSync(path.join(web, name)));
});
(async () => {
  fs.mkdirSync(output, {recursive: true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking']});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}, reducedMotion: 'reduce'});
    const page = await watch(context);
    await choose(page, 'synthetic-before-auth.mov', mov, 'video/quicktime');
    await page.locator('#manual-review > summary').click();
    await until(() => page.locator('#manual-export').isEnabled(), 'Local SHA was not computed');
    assert.equal(await page.locator('#analyze-button').isDisabled(), true);
    assert.equal(await page.locator('#manual-group').isEnabled(), true);
    assert.equal(await page.locator('#manual-applicability').inputValue(), 'unknown');
    const beforeAuth = await downloaded(page);
    assert.equal(beforeAuth.source.sha256, hash(mov));
    assert.equal(beforeAuth.is_ground_truth, false);
    assert.deepEqual(beforeAuth.marks, []);
    passed('Manual journal and real SHA export work before API authentication/model');

    await page.locator('#api-key').fill(testKey);
    await page.locator('#connect-button').click();
    await page.locator('#empty-result h3').filter({hasText: 'Локальная модель не настроена'}).waitFor();
    const networkStart = requests.length;
    await choose(page, 'synthetic-a.mp4', mp4);
    await until(async () => await page.locator('#manual-export').isEnabled() && await page.locator('#manual-add-mark').isEnabled(), 'Synthetic VP9 video metadata not decoded');
    await page.locator('#manual-group').fill('synthetic-scene-group');
    await page.locator('#manual-camera').fill('synthetic-camera');
    await page.locator('#manual-note').fill('<img src=x onerror=window.manualXss=true> literal synthetic note');
    await page.evaluate(() => { document.querySelector('#source-video').currentTime = 1.25; });
    await until(() => page.locator('#manual-time').innerText().then(text => text.includes('1,25')), 'Timecode not updated');
    await page.locator('#manual-mark-note').fill('Synthetic timestamp note');
    await page.locator('#manual-add-mark').click();
    await page.locator('.manual-rule-section > summary').click();
    await page.locator('#manual-rule-id').fill('synthetic_rule');
    await page.locator('#manual-rule-note').fill('<script>window.manualXss=true</script>');
    await page.locator('#manual-save-rule').click();
    const result = await downloaded(page);
    assert.deepEqual(Object.keys(result).sort(), ['camera_id', 'group_id', 'is_ground_truth', 'kind', 'marks', 'note', 'rule_reviews', 'schema_version', 'source']);
    assert.equal(result.schema_version, 1);
    assert.equal(result.kind, 'cashwatch_manual_review');
    assert.equal(result.is_ground_truth, false);
    assert.equal(result.source.sha256, hash(mp4));
    assert.equal(result.source.size_bytes, mp4.length);
    assert.equal(result.source.duration_sec, 3);
    assert.equal(result.source.filename, 'synthetic-a.mp4');
    assert.equal(result.marks[0].timestamp_sec, 1.25);
    assert.equal(result.rule_reviews[0].applicability, 'unknown');
    assert.equal(result.group_id, 'synthetic-scene-group');
    assert.equal(JSON.stringify(result).includes(testKey), false);
    assert.equal(await page.evaluate(() => Boolean(window.manualXss)), false);
    assert.equal(await page.locator('#manual-rules script').count(), 0);
    passed('Real decoded synthetic timestamp, unknown applicability, exact JSON contract and text escaping');

    await page.locator('#manual-rule-id').fill('synthetic_rule');
    await page.locator('#manual-applicability').selectOption('confirmed');
    await page.locator('#manual-export').click();
    assert.match(await page.locator('#manual-status').innerText(), /Сначала сохраните/);
    await page.locator('#manual-save-rule').click();
    assert.equal((await downloaded(page)).rule_reviews[0].applicability, 'confirmed');
    await page.locator('#manual-rule-id').fill('synthetic_rule');
    await page.locator('#manual-applicability').selectOption('not_applicable');
    await page.locator('#manual-save-rule').click();
    const updated = await downloaded(page);
    assert.equal(updated.rule_reviews.length, 1);
    assert.equal(updated.rule_reviews[0].applicability, 'not_applicable');
    passed('Unsaved rule draft blocks export; explicit applicability updates replace same rule once');

    await choose(page, 'synthetic-b.mp4', second);
    await until(() => page.locator('#manual-export').isEnabled(), 'Second SHA missing');
    assert.equal(await page.locator('#manual-group').inputValue(), '');
    await page.locator('#manual-group').fill('separate-second-group');
    await choose(page, 'synthetic-a-alias.mp4', mp4);
    await until(() => page.locator('#manual-group').inputValue().then(value => value === 'synthetic-scene-group'), 'Same SHA draft not restored');
    assert.equal(await page.locator('#manual-marks li').count(), 1);
    const restored = await downloaded(page);
    assert.equal(restored.source.filename, 'synthetic-a-alias.mp4');
    assert.equal(restored.source.sha256, hash(mp4));
    passed('Different files isolated; same SHA under another filename restores edits and correct source');

    await page.locator('#manual-marks .manual-item > button').click();
    await page.locator('#manual-rules .manual-item > button').click();
    const deleted = await downloaded(page);
    assert.deepEqual(deleted.marks, []);
    assert.deepEqual(deleted.rule_reviews, []);
    await page.locator('#manual-mark-note').fill('Unattached synthetic note');
    await page.locator('#manual-export').click();
    assert.match(await page.locator('#manual-status').innerText(), /Добавьте временную отметку/);
    await page.locator('#manual-mark-note').fill('');
    passed('Deletion changes export; an unattached mark note cannot silently disappear');

    for (const [width, height] of [[1440, 1000], [390, 844], [320, 740]]) {
      await page.setViewportSize({width, height});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
      await page.screenshot({path: path.join(output, `manual-${width}.png`), fullPage: true, animations: 'disabled'});
    }
    passed('Expanded manual panel fits desktop, 390px and 320px layouts');
    assert.deepEqual(requests.slice(networkStart).filter(req => req.url.startsWith('http')), []);
    assert.deepEqual(await page.evaluate(() => ({local: localStorage.length, session: sessionStorage.length})), {local: 0, session: 0});
    passed('Selection, editing and export make zero HTTP requests and use no persistent browser storage');

    const race = await browser.newContext();
    await race.addInitScript(() => {
      const digest = crypto.subtle.digest.bind(crypto.subtle); let calls = 0;
      crypto.subtle.digest = async (...args) => { if (++calls === 1) await new Promise(resolve => setTimeout(resolve, 450)); return digest(...args); };
    });
    const racePage = await watch(race);
    await choose(racePage, 'synthetic-slow-a.mov', mov, 'video/quicktime');
    await choose(racePage, 'synthetic-fast-b.mov', second, 'video/quicktime');
    await racePage.locator('#manual-review > summary').click();
    await until(() => racePage.locator('#manual-export').isEnabled(), 'Second selection SHA missing');
    await racePage.locator('#manual-group').fill('fast-file-preserved');
    await new Promise(resolve => setTimeout(resolve, 600));
    const raceResult = await downloaded(racePage);
    assert.equal(raceResult.source.sha256, hash(second));
    assert.equal(raceResult.source.filename, 'synthetic-fast-b.mov');
    assert.equal(raceResult.group_id, 'fast-file-preserved');
    passed('Delayed real Web Crypto digest cannot overwrite a newer file or its notes');

    const failed = await browser.newContext();
    await failed.addInitScript(() => { crypto.subtle.digest = async () => { throw new Error('Synthetic hash failure'); }; });
    const failedPage = await watch(failed);
    await choose(failedPage, 'synthetic-hash-failure.mov', mov, 'video/quicktime');
    await failedPage.locator('#manual-review > summary').click();
    await until(() => failedPage.locator('#manual-status').innerText().then(text => text.includes('Не удалось вычислить SHA-256')), 'Hash error not shown');
    assert.equal(await failedPage.locator('#manual-export').isDisabled(), true);
    assert.equal(await failedPage.locator('#file-panel').isVisible(), true);
    passed('Hash failure explicitly blocks export while preserving local preview');

    // Regression reported by Plus 2: a once-valid duration must not survive unavailable media metadata.
    const durationContext = await browser.newContext();
    const durationPage = await watch(durationContext);
    await durationPage.locator('#api-key').fill(testKey);
    await durationPage.locator('#connect-button').click();
    await durationPage.locator('#empty-result h3').filter({hasText: 'Локальная модель не настроена'}).waitFor();
    await choose(durationPage, 'synthetic-duration-regression.mp4', mp4);
    await durationPage.locator('#manual-review > summary').click();
    await until(() => durationPage.locator('#manual-add-mark').isEnabled(), 'Regression source did not decode');
    await durationPage.locator('#manual-note').fill('Preserve this synthetic manual review');
    await durationPage.evaluate(() => { document.querySelector('#source-video').currentTime = 1.25; });
    await until(() => durationPage.locator('#manual-time').innerText().then(text => text.includes('1,25')), 'Regression timecode missing');
    await durationPage.locator('#manual-mark-note').fill('Preserve this previously recorded synthetic timestamp');
    await durationPage.locator('#manual-add-mark').click();
    assert.equal((await downloaded(durationPage)).source.duration_sec, 3);
    for (const mode of ['unknown-duration', 'media-error', 'empty-metadata']) {
      await durationPage.evaluate(mode => {
        const video = document.querySelector('#source-video');
        if (mode === 'unknown-duration') Object.defineProperty(video, 'duration', {configurable: true, get: () => NaN});
        if (mode === 'media-error') Object.defineProperty(video, 'error', {configurable: true, get: () => ({code: 3, message: 'Synthetic media error'})});
        if (mode === 'empty-metadata') Object.defineProperty(video, 'readyState', {configurable: true, get: () => 0});
        video.dispatchEvent(new Event(mode === 'unknown-duration' ? 'durationchange' : mode === 'media-error' ? 'error' : 'emptied'));
      }, mode);
      const unknown = await downloaded(durationPage);
      assert.equal(unknown.source.duration_sec, null, `${mode}: stale duration was exported`);
      assert.equal(unknown.marks.length, 1, `${mode}: prior manual mark was lost`);
      assert.equal(unknown.marks[0].timestamp_sec, 1.25);
      assert.equal(unknown.note, 'Preserve this synthetic manual review');
      assert.equal(await durationPage.locator('#manual-add-mark').isDisabled(), true);
      assert.equal(await durationPage.locator('#manual-time').innerText(), '—');
      assert.equal((await durationPage.locator('#file-info').innerText()).includes('0:03'), false);
      assert.match(await durationPage.locator('#manual-status').innerText(), /отметки.*проверки/);
      await durationPage.locator('#manual-marks .manual-item-copy > button').click();
      assert.match(await durationPage.locator('#source-message').innerText(), /Браузер пока не может перейти/);
      await durationPage.evaluate(() => {
        const video = document.querySelector('#source-video');
        delete video.duration; delete video.error; delete video.readyState;
        video.dispatchEvent(new Event('loadedmetadata'));
      });
      assert.equal((await downloaded(durationPage)).source.duration_sec, 3);
    }
    passed('Invalidated duration, media error and emptied metadata export null, preserve prior marks and block time actions');

    // The aggregate model flag must not hide the explicit cause returned by the backend.
    const readinessCases = [
      {name: 'Missing weights take priority over aggregate model=false and other missing components',
        fields: {model_configured: false, weights_present: false, ml_packages_present: false, decoders_present: false, active_rules: 0}, title: 'Веса модели не найдены'},
      {name: 'Missing ML packages take priority over aggregate model=false',
        fields: {model_configured: false, weights_present: true, ml_packages_present: false, decoders_present: true}, title: 'Среда анализа не готова'},
      {name: 'Missing decoder takes priority over aggregate model=false',
        fields: {model_configured: false, weights_present: true, ml_packages_present: true, decoders_present: false}, title: 'Среда анализа не готова'},
      {name: 'Legacy unconfigured capabilities keep the generic model fallback',
        fields: {model_configured: false}, title: 'Локальная модель не настроена'},
      {name: 'Unknown optional fields do not become explicit missing-component failures',
        fields: {model_configured: false, weights_present: null, ml_packages_present: null, decoders_present: null}, title: 'Локальная модель не настроена'},
      {name: 'Configured model with no active rules keeps its separate rules blocker',
        fields: {model_configured: true, weights_present: true, ml_packages_present: true, decoders_present: true, active_rules: 0}, title: 'Правила требуют настройки'},
      {name: 'Legacy configured capabilities without optional flags still allow analysis',
        fields: {model_configured: true}, title: null},
    ];
    await page.setViewportSize({width: 1440, height: 1000});
    for (const scenario of readinessCases) {
      capabilities = {...fileCapabilities, ...scenario.fields};
      await page.locator('#api-key').fill(testKey);
      const response = page.waitForResponse(result => result.url().endsWith('/api/v1/capabilities'));
      await page.locator('#connect-button').click();
      assert.equal((await response).status(), 200);
      await until(() => page.locator('#connect-button').isEnabled(), 'Readiness check did not complete');
      if (scenario.title) {
        assert.equal(await page.locator('#empty-result h3').innerText(), scenario.title);
        assert.equal(await page.locator('#analyze-button').isDisabled(), true);
        assert.match(await page.locator('#connection-message').getAttribute('class'), /warning/);
      } else {
        assert.equal(await page.locator('#analyze-button').isEnabled(), true);
        assert.equal(await page.locator('#result-state').innerText(), 'Готово к запуску');
      }
      assert.equal(await page.locator('#manual-export').isEnabled(), true);
      assert.equal(await page.locator('#manual-group').inputValue(), 'synthetic-scene-group');
      passed(scenario.name);
    }
    await page.locator('#api-key').fill('synthetic-incorrect-api-key');
    const denied = page.waitForResponse(result => result.url().endsWith('/api/v1/capabilities'));
    await page.locator('#connect-button').click();
    assert.equal((await denied).status(), 401);
    await until(() => page.locator('#connect-button').isEnabled(), 'Auth failure did not complete');
    assert.match(await page.locator('#connection-message').innerText(), /Ключ доступа отсутствует или недействителен/);
    assert.equal(await page.locator('#analyze-button').isDisabled(), true);
    assert.equal(await page.locator('#manual-export').isEnabled(), true);
    passed('HTTP 401 remains a separate auth failure and does not block the local journal');
    await page.locator('#api-key').fill(testKey);
    await page.locator('#connect-button').click();
    await until(() => page.locator('#analyze-button').isEnabled(), 'Reauthentication did not complete');
    page.on('dialog', dialog => dialog.accept());
    await page.reload({waitUntil: 'networkidle'});
    assert.equal(await page.locator('#api-key').inputValue(), '');
    assert.equal(await page.locator('#analyze-button').isDisabled(), true);
    assert.deepEqual(await page.evaluate(() => ({local: localStorage.length, session: sessionStorage.length})), {local: 0, session: 0});
    passed('Reload clears the key and authorization without using persistent browser storage');
    assert.deepEqual(errors, []);
    assert.equal(requests.some(req => req.method !== 'GET' || (req.url.startsWith('http') && !req.url.startsWith(origin + '/'))), false);
    passed('No JavaScript exceptions, uploads or external requests in all focused cases');
    fs.writeFileSync(path.join(output, 'manual-review-report.json'), JSON.stringify({browser: browser.version(), mode: 'real UI/static local harness; synthetic health/capabilities, synthetic media; no model', results}, null, 2) + '\n');
    process.stdout.write(JSON.stringify({passed: results.length, failed: 0, output}) + '\n');
  } finally { await browser.close(); server.close(); }
})().catch(error => { server.close(); process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
