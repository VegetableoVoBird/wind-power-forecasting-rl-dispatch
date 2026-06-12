import { createRouter, createWebHistory } from 'vue-router'
import OverviewPage from './views/OverviewPage.vue'
import SitesPage from './views/SitesPage.vue'
import PredictionPage from './views/PredictionPage.vue'
import ExperimentsPage from './views/ExperimentsPage.vue'
import AgentPage from './views/AgentPage.vue'

const routes = [
  { path: '/', redirect: '/overview' },
  { path: '/overview', name: 'overview', component: OverviewPage },
  { path: '/prediction', name: 'prediction', component: PredictionPage },
  { path: '/sites', name: 'sites', component: SitesPage },
  { path: '/experiments', name: 'experiments', component: ExperimentsPage },
  { path: '/agent', name: 'agent', component: AgentPage },
]

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  },
})
