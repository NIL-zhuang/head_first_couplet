import Couplet from "@/views/Couplet.vue";
import Home from "@/views/Home.vue";
import Panel from "@/views/Panel.vue";
import Poem from "@/views/Poem.vue";
import { createRouter, createWebHistory } from "vue-router";

const routes = [
  {
    path: "/",
    name: "home",
    component: Home,
  },
  {
    path: "/panel",
    name: "panel",
    component: Panel,
    redirect: (to) => ({ name: "couplet" }),
    children: [
      {
        path: "/poem",
        name: "poem",
        component: Poem,
      },
      {
        path: "/couplet",
        name: "couplet",
        component: Couplet,
      },
    ],
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});
export default router;
