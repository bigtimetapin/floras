package discretizer

object Audius extends App {

  def apply(
    leader: String,
    target: String,
    m: List[(String, String)],
    path: List[(String, String)]
  ): List[List[(String, String)]] = {
    m.flatMap { tup =>
      if (tup._1 == leader) {
        if (tup._2 == target) {
          // end
          List(tup +: path)
        } else {
          // traverse
          apply(
            leader = tup._2,
            target = target,
            m = m.filterNot(_.eq(tup)),
            path = tup +: path
          )
        }
      } else if (tup._2 == leader) {
        if (tup._1 == target) {
          // end
          List(tup +: path)
        } else {
          // traverse
          apply(
            leader = tup._1,
            target = target,
            m = m.filterNot(_.eq(tup)),
            path = tup +: path
          )
        }
      } else {
        // dead end
        Nil
      }
    }
  }

  val m = List(
    ("a", "b"),
    ("c", "b"),
    ("d", "e"),
    ("a", "d"),
    ("c", "d"),
    ("f", "g")
  )

  println(apply(leader = "e", target = "a", m = m, path = Nil))
  println(apply(leader = "a", target = "e", m = m, path = Nil))
  println(apply(leader = "a", target = "d", m = m, path = Nil))
  println(apply(leader = "a", target = "c", m = m, path = Nil))
  println(apply(leader = "a", target = "b", m = m, path = Nil))
  println(apply(leader = "f", target = "g", m = m, path = Nil))
  println(apply(leader = "f", target = "a", m = m, path = Nil))

}
