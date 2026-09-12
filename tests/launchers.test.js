const test = require('node:test')
const assert = require('node:assert/strict')
const launcher = require('../pinokio')
const install = require('../install')
const start = require('../start')
const reset = require('../reset')
const update = require('../update')
const torch = require('../torch')

const menu = (files = [], running = [], local = {}) => launcher.menu(null, {
  exists: path => files.includes(path),
  running: path => running.includes(path),
  local: () => local
})

test('stale flags cannot expose Start and failed installs can be reset', async () => {
  assert.equal((await menu(['installed.flag']))[0].text, 'Install')
  assert.equal((await menu(['app/env']))[1].href, 'reset.js')
  assert.equal((await menu(['app/env', 'installed.flag']))[0].text, 'Start')
})

test('maintenance stays visible after the install flag disappears', async () => {
  for (const name of ['install', 'update', 'reset']) {
    const items = await menu([], [`${name}.js`])
    assert.equal(items[0].href, `${name}.js`)
    assert.equal(items[0].default, true)
  }
})

test('running app defaults to captured URL only when ready', async () => {
  const files = ['app/env', 'installed.flag']
  assert.equal((await menu(files, ['start.js']))[0].href, 'start.js')
  assert.equal((await menu(files, ['start.js'], { url: 'http://127.0.0.1:7861' }))[0].href, 'http://127.0.0.1:7861')
})

test('URL capture ignores external links and returns the local URL in group 1', () => {
  const pattern = start.run[0].params.on[0].event
  const regex = new RegExp(pattern.slice(1, -1))
  assert.equal(regex.exec('Read http://example.com/help'), null)
  assert.equal(regex.exec('Running on local URL:  http://127.0.0.1:7861')[1], 'http://127.0.0.1:7861')
  assert.equal(start.run[1].params.url, '{{input.event[1]}}')
})

test('install and reset invalidate completion before changing the environment', () => {
  for (const script of [install, reset]) {
    assert.equal(script.run[0].method, 'fs.rm')
    assert.equal(script.run[0].params.path, 'installed.flag')
  }
  assert.equal(install.run.at(-2).params.message, 'uv pip check')
  assert.equal(install.run.at(-1).method, 'fs.write')
  assert.equal(update.run[0].params.message, 'git pull --ff-only')
  assert.equal(update.run[1].params.uri, 'install.js')
})

test('install preloads Perth build dependencies before disabling isolation', () => {
  const commands = install.run[1].params.message
  assert.match(commands[1], /uv_build~=0\.12\.7/)
  assert.match(commands[2], /--no-build-isolation/)
})

test('platform routing selects compatible torch builds with dependencies', () => {
  const cases = [
    ['win32', 'x64', 'nvidia', 'cu124'],
    ['linux', 'x64', 'nvidia', 'cu124'],
    ['linux', 'x64', 'amd', 'rocm6.2.4'],
    ['darwin', 'arm64', 'apple', 'pypi.org/simple'],
    ['win32', 'x64', 'amd', '/cpu'],
    ['linux', 'x64', null, '/cpu']
  ]
  for (const [platform, arch, gpu, index] of cases) {
    const step = torch.run.find(step => !step.when ||
      new Function('platform', 'arch', 'gpu', `return ${step.when.slice(2, -2)}`)(platform, arch, gpu))
    assert.ok(step.params.message.includes(index))
    assert.ok(step.params.message.includes('torch==2.6.0'))
    assert.ok(step.params.message.includes('torchaudio==2.6.0'))
    assert.ok(!step.params.message.includes('--no-deps'))
    assert.equal(step.next, null)
  }
})
