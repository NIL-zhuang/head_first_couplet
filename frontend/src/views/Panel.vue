<template>
    <div>
        <nav class="sidebar">
            <header>
                <div>AI Poet & Couplet</div>
            </header>
            <div>
                <ul>
                    <router-link to="/">
                        <li class="nav-link">
                            <div class="bx bx-home-alt icon" to="/">
                                <span class="nav-text"> Home </span>
                            </div>
                        </li>
                    </router-link>
                    <router-link to="/poem">
                        <li class="nav-link">
                            <div class="bx bx-heart icon">
                                <span class="nav-text"> Poem </span>
                            </div>
                        </li>
                    </router-link>
                    <router-link to="/couplet">
                        <li class="nav-link">
                            <div class="bx bx-heart icon">
                                <span class="nav-text"> Couplet </span>
                            </div>
                        </li>
                    </router-link>
                </ul>
            </div>
        </nav>
        <router-view></router-view>
    </div>
</template>

<style scoped>
@import "@/assets/style/panel.css";
</style>