<template>
    <div class="main">
        <div class="component">
            <div class="left-input">
                <div class="title">AI Poem</div>
                <div class="hint">请输入想模仿的诗人</div>
                <input type="text" v-model="author" placeholder="作者，如李白，也可以不填">
                <div class="hint">诗句的标题</div>
                <input type="text" v-model="title" placeholder="标题，如静夜思" required="required">
                <button @click="getPoem">go</button>
            </div>
            <div class="right-output">
                <div class="hint">生成的诗句是：</div>
                <el-card class="box-card" shadow="always">
                    <el-empty v-if="result.length === 0" description="暂无数据"></el-empty>
                    <div v-else v-for="sentence in result" :key="sentence.id" class="output-text">
                        {{ sentence.text }}
                    </div>
                </el-card>
            </div>
        </div>
    </div>
</template>

<script setup>
import { getCurrentInstance, reactive, ref } from 'vue';
const internalInstance = getCurrentInstance();
const axios = internalInstance.appContext.config.globalProperties.axios;

const title = ref('')
const author = ref('')
let result = reactive([])

function getPoem() {
    if (title.value == '') {
        alert('请输入标题')
        return
    }
    axios.post('http://127.0.0.1:5000/poem', {
        'author': author.value,
        'title': title.value,
    }).then(async function (response) {
        let id = 0
        var item
        result.length = 0
        for (item in response.data) {
            result.push({
                id: id++,
                text: response.data[item]
            })
        }
        console.log(result)
    }).catch(function (error) {
        console.log(error);
    })
}
</script>

<style scoped>
@import "@/assets/style/module.css";
</style>