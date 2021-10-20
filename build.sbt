scalaVersion := "2.12.8"
name := "floras"
organization := "in.tap"
version := "0.0.1-SNAPSHOT"

libraryDependencies ++= Seq(
  "org.bytedeco" % "javacv" % "1.4.4",
  "org.bytedeco" % "javacpp" % "1.4.4",
  "org.bytedeco.javacpp-presets" % "opencv-platform" % "4.0.1-1.4.4",
  "org.bytedeco.javacpp-presets" % "ffmpeg-platform" % "4.1-1.4.4"
)
