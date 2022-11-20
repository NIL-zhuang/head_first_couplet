<template>
    <div class="main">
        <div class="component">
            <div class="left-input">
                <div class="title">AI Couplet</div>
                <div class="hint">请输入对联的上联</div>
                <input type="text" v-model="preCouplet" placeholder="上联，如灵蛇出洞千山秀">
                <button @click="getNextCouplet"> go </button>
            </div>
            <div class="right-output">
                <div class="hint">AI提供的对联下联是：</div>
                <el-card class="box-card" shadow="always">
                    <div class="output-text">{{ nextCouplet }}</div>
                </el-card>
            </div>
        </div>
    </div>
</template>

<script setup>
import { getCurrentInstance, ref } from 'vue';

const preCouplet = ref()
const nextCouplet = ref('骏马踏春万木荣')
const internalInstance = getCurrentInstance();
const axios = internalInstance.appContext.config.globalProperties.axios;

function getNextCouplet() {
    console.log('preCouplet: ', preCouplet.value)
    axios.post('http://127.0.0.1:5000/couplet', {
        'upper': preCouplet.value,
    }).then(function (response) {
        console.log(response.data)
        nextCouplet.value = response.data
    }).catch(function (error) {
        console.log(error);
    })
}
</script>

<style scoped>
@import "@/assets/style/module.css";
</style>