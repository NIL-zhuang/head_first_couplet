<template>
    <div>
        <nav class="sidebar">
            <header>
                <div>AI Poet</div>
            </header>
            <div>
                <ul>
                    <li class="nav-link">
                        <router-link class="bx bx-home-alt icon" to="/">
                            <span class="nav-text"> Home </span>
                        </router-link>
                    </li>
                    <li class="nav-link">
                        <router-link class="bx bx-heart icon" to="/poem">
                            <span class="nav-text"> Poem </span>
                        </router-link>
                    </li>
                    <li class="nav-link">
                        <router-link class="bx bx-heart icon" to="/couplet">
                            <span class="nav-text"> Couplet </span>
                        </router-link>
                    </li>
                </ul>
            </div>
        </nav>
        <router-view></router-view>
    </div>
</template>

<style scoped>
@import "@/assets/style/panel.css";
</style>