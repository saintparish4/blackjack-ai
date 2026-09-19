// Regenerates docs/images/web_coach.png and web_play.png from the live page.
// Optional tooling, not needed to build or test the demo:
//   npm i --no-save playwright && npx playwright install chromium
//   python3 -m http.server -d web 8765 &   # from the repo root
//   node web/test/screenshots.cjs
// Seed 2026 deals the same shoe every time, so the captures are reproducible.
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('http://localhost:8765/?seed=2026');
  await page.waitForSelector('#loading', { state: 'hidden' });

  // Coach mode on a doubling decision.
  await page.click('button[data-mode="coach"]');
  for (let i = 0; i < 200; i++) {
    await page.click('#deal');
    const sit = await page.textContent('#ai-body .situation').catch(() => '');
    const pick = await page.textContent('#ai-body .pick strong').catch(() => '');
    if (/hard 11 vs [2-6]/.test(sit) && pick === 'Double') break;
    // finish the hand the way the coach says so Deal re-enables
    while (await page.$('#actions button.suggested')) await page.click('#actions button.suggested');
  }
  await page.mouse.move(0, 0); await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(400);
  await page.screenshot({ path: require('path').join(__dirname, '../../docs/images/web_coach.png') });

  // Play mode: a visitor stands on 12 against a strong card, then sees the verdict.
  await page.click('#actions button.suggested');
  await page.click('button[data-mode="play"]');
  for (let i = 0; i < 300; i++) {
    await page.click('#deal');
    const you = await page.textContent('#hands .seat-label').catch(() => '');
    const dealer = await page.textContent('#dealer-total');
    if (/\((12|13|14)\)/.test(you) && /showing (9|10|A|11)/.test(dealer)) {
      await page.click('#actions button:has-text("Stand")');
      break;
    }
    while (await page.$('#actions button:not([disabled])')) await page.click('#actions button:has-text("Stand")');
  }
  await page.mouse.move(0, 0); await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(400);
  await page.screenshot({ path: require('path').join(__dirname, '../../docs/images/web_play.png') });
  console.log(errors.length ? errors : 'ok');
  await browser.close();
})();
