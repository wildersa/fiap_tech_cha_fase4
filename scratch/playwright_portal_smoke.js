const { chromium } = require('playwright');

const baseUrl = process.env.PORTAL_URL || 'http://127.0.0.1:8000/dashboard';

function messageBody(message) {
  return {
    type: message.type(),
    text: message.text(),
    location: message.location(),
  };
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleMessages = [];
  const failedRequests = [];
  const badResponses = [];

  page.on('console', message => {
    if (['error', 'warning'].includes(message.type())) {
      consoleMessages.push(messageBody(message));
    }
  });
  page.on('pageerror', error => {
    consoleMessages.push({ type: 'pageerror', text: error.message, location: {} });
  });
  page.on('requestfailed', request => {
    failedRequests.push({
      url: request.url(),
      method: request.method(),
      failure: request.failure()?.errorText,
    });
  });
  page.on('response', response => {
    const status = response.status();
    const url = response.url();
    if (status >= 400 && !url.includes('/.well-known/appspecific/')) {
      badResponses.push({ url, status, method: response.request().method() });
    }
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForSelector('#tab-inference', { timeout: 10000 });
  await page.waitForTimeout(1000);

  const title = await page.title();
  const config = await page.evaluate(() => window.DASHBOARD_CONFIG || null);
  const hasEnableReferenceError = consoleMessages.some(item => item.text.includes('__ENABLE_TRAINING_API__'));
  const hasFeatureRegistryError = consoleMessages.some(item => item.text.includes('FEATURE_REGISTRY'));

  await page.click('#tab-inference');
  await page.waitForSelector('#payload', { timeout: 10000 });
  const initialPayload = JSON.parse(await page.locator('#payload').inputValue());
  const initialPayloadRows = Array.isArray(initialPayload.closes)
    ? initialPayload.closes.length
    : (initialPayload.rows || []).length;

  const predictBefore = await page.locator('#predictedClose').textContent();
  const predictResponsePromise = page.waitForResponse(
    response => response.url().includes('/predict') && response.request().method() === 'POST',
    { timeout: 30000 }
  ).catch(error => ({ error: error.message }));
  await page.getByRole('button', { name: /Executar Inferência/i }).click();
  const predictResponse = await predictResponsePromise;
  await page.waitForTimeout(1000);
  const predictAfter = await page.locator('#predictedClose').textContent();
  const predictionError = await page.locator('#prediction-error').isVisible().catch(() => false)
    ? await page.locator('#prediction-error').textContent()
    : '';

  await page.click('#tab-telemetry');
  await page.waitForSelector('#tel-reqs', { timeout: 10000 });
  await page.waitForTimeout(1000);
  const telemetry = {
    inferences: await page.locator('#tel-reqs').textContent(),
    latency: await page.locator('#tel-lat').textContent(),
    chartCanvas: await page.locator('#resourceChart').count(),
  };

  const trainTabCount = await page.locator('#tab-train').count();
  if (trainTabCount) {
    await page.click('#tab-train');
    await page.waitForTimeout(1500);
  }
  const trainVisible = await page.locator('#train').isVisible().catch(() => false);
  const runRowsBeforeSearch = await page.locator('#runs-table-body tr:not([id^="details-"])').count();
  const trophyCount = await page.locator('#runs-table-body td >> text=🏆').count();
  const firstRunDate = runRowsBeforeSearch
    ? await page.locator('#runs-table-body tr:not([id^="details-"])').first().locator('td').nth(2).textContent()
    : '';
  await page.locator('#runs-search').fill('bb051a12');
  await page.waitForTimeout(300);
  const runRowsAfterSearch = await page.locator('#runs-table-body tr:not([id^="details-"])').count();
  await page.locator('#runs-search').fill('');
  await page.locator('th', { hasText: 'Ganho MAPE' }).click();
  await page.waitForTimeout(300);
  const gainSortIndicator = await page.locator('#sort-baseline_gain_pct').textContent();
  const fillButtons = await page.getByRole('button', { name: /Preencher Formulário/i }).count();
  let fillFormResult = 'not_available';
  if (fillButtons > 0) {
    await page.getByRole('button', { name: /Preencher Formulário/i }).first().click();
    await page.waitForTimeout(1000);
    fillFormResult = await page.locator('#lineage-badge').isVisible().catch(() => false)
      ? 'filled_with_lineage'
      : 'clicked';
  }

  const result = {
    url: page.url(),
    title,
    config,
    initialPayloadRows,
    predictBefore,
    predictAfter,
    predictResponse: predictResponse?.error
      ? predictResponse
      : { status: predictResponse.status(), url: predictResponse.url() },
    predictionError,
    telemetry,
    trainVisible,
    runRowsBeforeSearch,
    trophyCount,
    firstRunDate,
    runRowsAfterSearch,
    gainSortIndicator,
    fillButtons,
    fillFormResult,
    hasEnableReferenceError,
    hasFeatureRegistryError,
    consoleMessages,
    failedRequests,
    badResponses,
  };

  console.log(JSON.stringify(result, null, 2));
  await browser.close();
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
