import { useRef } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { motion } from 'framer-motion'
import MovieCard from './MovieCard'

function MovieRow({ title, movies, onPlay, onInfo }) {
  const rowRef = useRef(null)

  const scroll = (direction) => {
    if (rowRef.current) {
      const { scrollLeft, clientWidth } = rowRef.current
      const scrollTo =
        direction === 'left'
          ? scrollLeft - clientWidth
          : scrollLeft + clientWidth
      rowRef.current.scrollTo({ left: scrollTo, behavior: 'smooth' })
    }
  }

  if (!movies || movies.length === 0) {
    return null
  }

  return (
    <div className="mb-8">
      <h2 className="text-xl font-bold mb-4 px-8">{title}</h2>
      <div className="relative group">
        <button
          onClick={() => scroll('left')}
          className="absolute left-0 top-0 bottom-0 z-10 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity p-2"
        >
          <ChevronLeft className="w-8 h-8" />
        </button>
        <div
          ref={rowRef}
          className="flex space-x-4 overflow-x-auto scrollbar-hide px-8 scroll-smooth"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {movies.map((movie, index) => (
            <motion.div
              key={movie.movie_id || index}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex-shrink-0 w-48"
            >
              <MovieCard movie={movie} onPlay={onPlay} onInfo={onInfo} />
            </motion.div>
          ))}
        </div>
        <button
          onClick={() => scroll('right')}
          className="absolute right-0 top-0 bottom-0 z-10 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity p-2"
        >
          <ChevronRight className="w-8 h-8" />
        </button>
      </div>
    </div>
  )
}

export default MovieRow

