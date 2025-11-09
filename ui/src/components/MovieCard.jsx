import { useState } from 'react'
import { Play, Info } from 'lucide-react'
import { motion } from 'framer-motion'

function MovieCard({ movie, onPlay, onInfo }) {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      className="relative group cursor-pointer"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2 }}
    >
      <div className="relative overflow-hidden rounded-lg">
        <img
          src={movie.poster_url || '/placeholder-poster.jpg'}
          alt={movie.title}
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/300x450?text=No+Poster'
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        
        {isHovered && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center space-x-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <button
              onClick={() => onPlay && onPlay(movie)}
              className="bg-white text-black px-4 py-2 rounded-full flex items-center space-x-2 hover:bg-gray-200 transition-colors"
            >
              <Play className="w-5 h-5 fill-current" />
              <span>Play</span>
            </button>
            <button
              onClick={() => onInfo && onInfo(movie)}
              className="bg-gray-600/80 text-white px-4 py-2 rounded-full flex items-center space-x-2 hover:bg-gray-500 transition-colors"
            >
              <Info className="w-5 h-5" />
              <span>Info</span>
            </button>
          </motion.div>
        )}
      </div>
      
      <div className="mt-2">
        <h3 className="text-sm font-semibold truncate">{movie.title}</h3>
        {movie.rating && (
          <p className="text-xs text-netflix-gray">
            ⭐ {movie.rating.toFixed(1)}
          </p>
        )}
      </div>
    </motion.div>
  )
}

export default MovieCard

