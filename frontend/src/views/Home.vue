<template>
  <div id="home">
    <div class="center-in-center">
      <div class="title">
        <span class="text-light">AI Couplet &nbsp;&</span>
        <span class="text-dark">Poem Writing</span>
      </div>

      <div class="buttons">
        <router-link to="/couplet">
          <button id="button-trans">AI Couplet</button>
        </router-link>
        <router-link to="/poem">
          <button id="button-hard">AI Poem</button>
        </router-link>
      </div>
    </div>
  </div>
</template>

<style scoped>
@import "@/assets/style/style.css";

#home {
  height: 100vh;
  background: linear-gradient(-45deg, #fddf91, #e73c7e, #23a6d5, #23d5ab);
  background-size: 300% 300%;
  animation: gradient 10s ease infinite;
}

@keyframes gradient {
  0% {
    background-position: 0% 50%;
  }

  50% {
    background-position: 100% 50%;
  }

  100% {
    background-position: 0% 50%;
  }
}
</style>