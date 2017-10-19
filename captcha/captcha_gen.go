package main

import "github.com/afocus/captcha"
//import "fmt"
import "os"
import "image/png"
import "image/color"

func main() {
    capt := captcha.New()
	capt.SetFont("/home/zhangxiaojie/.fonts/COMIC.TTF")
    capt.SetDisturbance(captcha.MEDIUM)
    capt.SetFrontColor(color.RGBA{255, 0, 0, 255}, color.RGBA{0, 255, 0, 255}, color.RGBA{0, 0, 255, 255},
                       color.RGBA{0, 0, 0, 255})

    for i:=0; i< 100000; i++ {
	    img, str := capt.Create(6, captcha.ALL)
        //fmt.Println(str)
        f, err := os.Create("data/" + str + ".png")
        if err != nil {
            panic(err)
        }
        defer f.Close()
        png.Encode(f, img)
        f.Close()
    }
}
