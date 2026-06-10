const DEFAULT_BASE = '/api'
const rawBase = import.meta.env.VITE_API_BASE || DEFAULT_BASE
const API_BASE = rawBase.replace(/\/$/, '')

function buildApiUrl(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`
  if (normalized.startsWith('/api/')) {
    return `${API_BASE}${normalized.slice(4)}`
  }
  return `${API_BASE}${normalized}`
}

async function readJson(path, options = {}) {
  const response = await fetch(buildApiUrl(path), options)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `API ${path} failed: ${response.status}`)
  }
  return response.json()
}

export function fetchDashboard() {
  return readJson('/dashboard')
}

export function fetchSite(siteId) {
  return readJson(`/site/${siteId}`)
}

export function askAgent(question) {
  return readJson('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

export function refreshAgent() {
  return readJson('/agent/refresh', { method: 'POST' })
}

export function fetchComparison() {
  return readJson('/comparison')
}

export function getExportUrl(path) {
  return buildApiUrl(path)
}

// ---- 上传测试集管理 ----

export function saveUpload(csvContent, originalFilename) {
  return readJson('/upload/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ csv_content: csvContent, original_filename: originalFilename }),
  })
}

export function listUploads() {
  return readJson('/uploads')
}

export function getUploadDetail(uploadId) {
  return readJson(`/uploads/${uploadId}`)
}

export function getUploadSiteData(uploadId, siteId) {
  return readJson(`/uploads/${uploadId}/site/${siteId}`)
}

export function deleteUpload(uploadId) {
  return readJson(`/uploads/${uploadId}`, { method: 'DELETE' })
}
