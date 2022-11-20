import "@/assets/style/reset.css";
import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";

var app = createApp(App);

app.use(router).mount("#app");
