import axios from 'axios'

/**
 * axios 实例
 * - baseURL 为 /api：开发模式由 Vite 代理到 :8000，生产模式由 FastAPI 同源托管
 * - timeout 35s：略大于后端 30s 超时，确保后端先返回友好的 504，而非前端生硬掐断
 */
const http = axios.create({
  baseURL: '/api',
  timeout: 65000  // 略大于后端 60s 上限，避免后端仍在处理时前端先报错
})

/**
 * 智能问诊
 * @param {string} query 用户问题
 * @param {string} sessionId 会话 ID
 * @returns {Promise} 后端 ConsultationResponse：
 *   { answer, intent, symptoms, departments, medications,
 *     disclaimers, warnings, duration_ms, timestamp }
 */
export function consult(query, sessionId) {
  return http.post('/consult', { query, session_id: sessionId })
}

/**
 * 健康检查
 * @returns {Promise} { status, neo4j_connected, vector_index_loaded, agent_system_ready, ... }
 */
export function health() {
  return http.get('/health')
}

export default http
