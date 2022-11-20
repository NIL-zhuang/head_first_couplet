import "@/assets/style/reset.css";
import axios from "axios";
import { createApp } from "vue";
import VueAxios from "vue-axios";
import App from "./App.vue";
import router from "./router";

const app = createApp(App);
app.use(VueAxios, axios);

app.use(router).mount("#app");
