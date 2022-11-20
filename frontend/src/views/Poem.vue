<template>
    <div class="main">
        <div class="component">
            <div class="left-input">
                <div class="title">AI Poem</div>
                <div class="hint">请输入想模仿的诗人</div>
                <input type="text" v-model="author" placeholder="作者，如李白，也可以不填">
                <div class="hint">诗句的标题</div>
                <input type="text" v-model="title" placeholder="标题，如静夜思" required="required">
                <button>go</button>
            </div>
            <div class="right-output">
                <div class="hint">生成的诗句是：</div>
                <el-card class="box-card" shadow="always">
                    <div v-for="o in result" :key="result.id" class="output-text">{{ o.text }}</div>
                </el-card>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue';
let id = 0;
const title = ref('')
const author = ref('')
const result = ref([
    { id: id++, text: "床前明月光" },
    { id: id++, text: "疑是地上霜" },
    { id: id++, text: "举头望明月" },
    { id: id++, text: "低头思故乡" }
])
</script>

<style scoped>
@import "@/assets/style/module.css";
</style>