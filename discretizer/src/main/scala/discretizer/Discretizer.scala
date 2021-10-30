package discretizer

import org.bytedeco.javacv.{FFmpegFrameGrabber, Frame, Java2DFrameConverter}

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

object Discretizer extends App {

  private val frame2ImageConverter: Java2DFrameConverter = {
    new Java2DFrameConverter
  }

  private val _: Unit = {
    //apply("01")
    apply("02")
    apply("03")
  }

  private def apply(dir: String): Unit = {
    val frameGrabber: FFmpegFrameGrabber = {
      new FFmpegFrameGrabber(s"data/mp4/$dir/in.mp4")
    }
    val _: Unit = {
      frameGrabber
        .start()
    }
    // frame to image buffer conver
    // grab first frame
    var frame: Frame = {
      frameGrabber.grab()
    }
    // write until last frame
    while (frame != null) {
      val image: BufferedImage = {
        frame2ImageConverter.convert(frame)
      }
      ImageIO.write(
        image,
        "png",
        new File(s"data/png/$dir/${System.currentTimeMillis()}.png")
      )
      // grab next frame
      frame = frameGrabber.grab()
    }
    frameGrabber.stop()
  }

}
